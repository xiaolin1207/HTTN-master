import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import models
import data_got
import numpy as np
from utils import Bar, Logger, AverageMeter, precision_k,  ndcg_k, calc_f1, mkdir_p
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=45, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-c', '--checkpoint', default='imprint_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: imprint_checkpoint)')
parser.add_argument('--model', default='samp45_pretrain_checkpoint/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to model (default: none)')
parser.add_argument('--random', action='store_true', help='whether use random novel weights')
parser.add_argument('--num_sample', default=5, type=int,
                    metavar='N', help='number of novel sample (default: 1)')
parser.add_argument('--test-novel-only', action='store_true', help='whether only test on novel classes')
parser.add_argument('--aug', action='store_true', help='whether use data augmentation during training')
parser.add_argument('--lstm_hid_dim', default=150, type=int, metavar='N',
                    help='lstm_hid_dim')
parser.add_argument('--num_class', default=45, type=int, metavar='N',
                    help='the number of class')
parser.add_argument('--epochs', default=8 ,type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--samp_freq', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')


def main():
    global args, best_micro,base_sum
    args = parser.parse_args()

    base_transf, embed = data_got.one_sample_base2avg(batch_size=args.batch_size,sample_num=args.num_sample,samp_freq=args.samp_freq,num_class=args.num_class)
    Ftest_loader, novel_loader,novelall_loader,_ = data_got.Nload_data(batch_size=args.batch_size,sample_num=args.num_sample,num_class=args.num_class)
    embed = torch.from_numpy(embed).float()
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    model = models.Net(embed, args.lstm_hid_dim, num_classes=args.num_class).cuda()

    print('==> Reading from model checkpoint..')
    assert os.path.isfile(args.model), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model checkpoint '{}' (epoch {})"
          .format(args.model, checkpoint['epoch']))
    cudnn.benchmark = True
    real_weight=model.classifier.fc.weight.data

    criterion = nn.MSELoss()
    trans_model = models.Transfer().cuda()
    optimizer = torch.optim.Adam( trans_model.parameters(), lr=0.012, betas=(0.9, 0.99))


    for epoch in range(args.epochs):
        train_loss,base_sum= train(base_transf, trans_model,model, criterion,  optimizer,real_weight)
        print("loss",train_loss)
    imprint(novel_loader, model,trans_model, base_sum)
    model_criterion= nn.BCELoss()

    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    for i in range(2):
        print("ft start")
        fine_tuning(novelall_loader, model,  model_criterion, model_optimizer)
    validate(Ftest_loader, model)


def train(train_loader, trans_model,model, criterion,optimizer,real_weight):
    trans_model.train()
    base_rep=[]
    losses=[]
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(train_loader):
            input = input.cuda()
            output = model.extract(input)
            base_rep.extend(output.cpu().numpy())
    base_rep =np.array(base_rep)
    base_rep= torch.from_numpy(base_rep ).cuda()
    base_sum = torch.zeros(1350, 128).cuda()
    j = 0
    for i in range(1350):
        tmp =base_rep[j:j + args.num_sample]
        tmp = torch.sum(tmp, 0) / args.num_sample
        base_sum[i] = tmp / tmp.norm(p=2)
        j = j + args.num_sample
    e=0
    for h in range(args.samp_freq):
        doc_avg= base_sum[e:e+args.num_class,:]
        e=e+args.num_class
        output = trans_model(doc_avg)
        loss = criterion(output, real_weight)
        losses.append(float(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = np.mean(losses)

    return avg_loss,base_sum

def imprint(novel_loader, model, trans_model, base_sum):
    attention=[]
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(novel_loader):
            input = input.cuda()
            output = model.extract(input)
            if batch_idx == 0:
                output_stack = output
                target_stack = target
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, target), 0)
    new_weight = torch.zeros(9, 128).cuda()
    j=0
    for i in range(9):
        tmp=output_stack[j:j+args.num_sample]
        tmp=torch.sum(tmp,0)/args.num_sample
        new_weight[i] = tmp / tmp.norm(p=2)
        j=j+args.num_sample
    e=0
    for h in range(args.samp_freq):
        doc_avg = base_sum[e:e + args.num_class, :]
        pro = F.softmax(torch.mm(new_weight, doc_avg.t()))
        new_rep  = torch.mm(pro, doc_avg )
        attention.append(new_rep)
        e = e + args.num_class

    tail_corr = torch.zeros(9, 128).cuda()
    for m in range(args.samp_freq):
        tail_corr=tail_corr+attention[m]
    tail_corr=tail_corr/args.num_class
    print("attention is all")
    tail_rep = (tail_corr+new_weight)/2
    tail_real=trans_model.transfor( tail_rep)
    weight = torch.cat([model.classifier.fc.weight.data, tail_real])
    model.classifier.fc = nn.Linear(128, 54, bias=False)
    model.classifier.fc.weight.data = weight
    print('imprint done')

def validate(val_loader, model):
    F1 = np.zeros(54)
    score_micro = np.zeros(3)
    score_macro = np.zeros(3)
    test_p1, test_p3, test_p5 = 0, 0, 0
    test_ndcg1, test_ndcg3, test_ndcg5 = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(val_loader):
            # measure data loading time

            input = input.cuda()
            target = target.cuda()
            output = model(input)
            target = target.data.cpu().float()
            output = output.data.cpu()
            _p1, _p3, _p5 = precision_k(output.topk(k=5)[1].numpy(), target.numpy(), k=[1, 3, 5])
            test_p1 += _p1
            test_p3 += _p3
            test_p5 += _p5

            _ndcg1, _ndcg3, _ndcg5 = ndcg_k(output.topk(k=5)[1].numpy(), target.numpy(), k=[1, 3, 5])
            test_ndcg1 += _ndcg1
            test_ndcg3 += _ndcg3
            test_ndcg5 += _ndcg5
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
            for l in range(54):
                F1[l] += f1_score(target[:, l], output[:, l], average='binary')
                # precision[l] += precision_score(target[:, l], output[:, l], average='binary')
                # recall[l] += recall_score(target[:, l], output[:, l], average='binary')
            # micro, macro = calc_f1(target, output)
            # acc += accuracy_score(target, output)
            # print("acc",acc)
            score_micro += [precision_score(target, output, average='micro'),
                            recall_score(target, output, average='micro'),
                            f1_score(target, output, average='micro')]
            score_macro += [precision_score(target, output, average='macro'),
                            recall_score(target, output, average='macro'),
                            f1_score(target, output, average='macro')]
            # acc = calc_acc(target, output)
        np.set_printoptions(formatter={'float': '{: 0.4}'.format})
        print('the result of F1: \n', F1 / len(val_loader))
        print('the result of micro: \n', score_micro / len(val_loader))
        print('the result of macro: \n', score_macro / len(val_loader))
        test_p1 /= len(val_loader)
        test_p3 /= len(val_loader)
        test_p5 /= len(val_loader)

        test_ndcg1 /= len(val_loader)
        test_ndcg3 /= len(val_loader)
        test_ndcg5 /= len(val_loader)

        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (test_p1, test_p3, test_p5))
        print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg1, test_ndcg3, test_ndcg5))


def fine_tuning(train_loader, model, criterion, optimizer):
    F1 = np.zeros(54)
    score_micro = np.zeros(3)
    score_macro = np.zeros(3)
    data_time = AverageMeter()
    losses = AverageMeter()
    microF1 = AverageMeter()
    macroF1 = AverageMeter()
    model.train()
    test_p1, test_p3, test_p5 = 0, 0, 0
    test_ndcg1, test_ndcg3, test_ndcg5 = 0, 0, 0

    end = time.time()
    # bar = Bar('Training', max=len(train_loader))
    for batch_idx, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target.float())
        target = target.data.cpu().float()
        output=output.data.cpu()

        micro,macro = calc_f1( target,  output)
        losses.update(loss.item(), input.size(0))
        microF1.update(micro.item(), input.size(0))
        macroF1.update(macro.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        _p1, _p3, _p5 = precision_k(output.topk(k=5)[1].numpy(), target.numpy(), k=[1, 3, 5])
        test_p1 += _p1
        test_p3 += _p3
        test_p5 += _p5

        _ndcg1, _ndcg3, _ndcg5 = ndcg_k(output.topk(k=5)[1].numpy(), target.numpy(), k=[1, 3, 5])
        test_ndcg1 += _ndcg1
        test_ndcg3 += _ndcg3
        test_ndcg5 += _ndcg5
        output[output > 0.5] = 1
        output[output <= 0.5] = 0
        for l in range(54):
            F1[l] += f1_score(target[:, l], output[:, l], average='binary')
            # precision[l] += precision_score(target[:, l], output[:, l], average='binary')
            # recall[l] += recall_score(target[:, l], output[:, l], average='binary')
        # micro, macro = calc_f1(target, output)
        # acc += accuracy_score(target, output)
        # print("acc",acc)
        score_micro += [precision_score(target, output, average='micro'),
                        recall_score(target, output, average='micro'),
                        f1_score(target, output, average='micro')]
        score_macro += [precision_score(target, output, average='macro'),
                        recall_score(target, output, average='macro'),
                        f1_score(target, output, average='macro')]
        # acc = calc_acc(target, output)
    np.set_printoptions(formatter={'float': '{: 0.4}'.format})
    print('the result of F1: \n', F1 / len(train_loader))
    print('the result of micro: \n', score_micro / len(train_loader))
    print('the result of macro: \n', score_macro / len(train_loader))
    test_p1 /= len(train_loader)
    test_p3 /= len(train_loader)
    test_p5 /= len(train_loader)

    test_ndcg1 /= len(train_loader)
    test_ndcg3 /= len(train_loader)
    test_ndcg5 /= len(train_loader)

    print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (test_p1, test_p3, test_p5))
    print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg1, test_ndcg3, test_ndcg5))


if __name__ == '__main__':
    main()
