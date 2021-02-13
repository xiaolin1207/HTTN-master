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
from utils import Bar, Logger, AverageMeter, precision_k, calc_acc, ndcg_k, calc_f1, mkdir_p
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=90, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-c', '--checkpoint', default='imprint_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: imprint_checkpoint)')
parser.add_argument('--model', default='samp36_pretrain_checkpoint/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to model (default: none)')
parser.add_argument('--random', action='store_true', help='whether use random novel weights')
parser.add_argument('--num_sample', default=5, type=int,
                    metavar='N', help='number of novel sample (default: 1)')
parser.add_argument('--test-novel-only', action='store_true', help='whether only test on novel classes')
parser.add_argument('--aug', action='store_true', help='whether use data augmentation during training')
parser.add_argument('--lstm_hid_dim', default=150, type=int, metavar='N',
                    help='lstm_hid_dim')
parser.add_argument('--num_class', default=36, type=int, metavar='N',
                    help='the number of class')
parser.add_argument('--epochs', default=7 ,type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--samp_freq', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate')


def main():
    global args, best_micro
    args = parser.parse_args()

    base_transf, embed = data_got.one_sample_base2avg(batch_size=args.batch_size,sample_num=args.num_sample,samp_freq=args.samp_freq,num_class=args.num_class)
    Ftest_loader, novel_loader,novelall_loader= data_got.Nload_data(batch_size=args.batch_size,sample_num=args.num_sample,num_class=args.num_class)
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
    optimizer = torch.optim.Adam( trans_model.parameters(), lr=0.01, betas=(0.9, 0.99))

    for epoch in range(args.epochs):
        train_loss= train(base_transf, trans_model,model, criterion,  optimizer,real_weight)
        print("loss",train_loss)
    imprint(novel_loader, model,trans_model)
    # model_criterion= nn.BCELoss()
    microF1 = validate(Ftest_loader, model)
    # model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # for i in range(2):
    #     print("fine-tuning")
    #     train_loss, trn_micro, trn_macro = fine_tuning(novelall_loader, model,  model_criterion, model_optimizer)
    #     microF1 = validate(Ftest_loader, model)
    # print("microF1 and macroF1",microF1,macroF1)


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
    new_weight = torch.zeros(1350, 128).cuda()
    j = 0
    for i in range(1350):
        tmp =base_rep[j:j + args.num_sample]
        tmp = torch.sum(tmp, 0) / args.num_sample
        new_weight[i] = tmp / tmp.norm(p=2)
        j = j + args.num_sample
    e=0
    for h in range(args.samp_freq):
        doc_avg=new_weight[e:e+args.num_class,:]
        e=e+args.num_class
        output = trans_model(doc_avg)
        loss = criterion(output, real_weight)
        losses.append(float(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = np.mean(losses)

    return avg_loss

def imprint(novel_loader, model, trans_model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(novel_loader):
            data_time.update(time.time() - end)
            input = input.cuda()
            output = model.extract(input)
            if batch_idx == 0:
                output_stack = output
                target_stack = target
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, target), 0)
            batch_time.update(time.time() - end)
            end = time.time()
    new_weight = torch.zeros(9, 128).cuda()
    j=0
    for i in range(9):
        tmp=output_stack[j:j+args.num_sample]
        tmp=torch.sum(tmp,0)/args.num_sample
        new_weight[i] = tmp / tmp.norm(p=2)
        j=j+args.num_sample

    tail_real=trans_model.transfor(new_weight)
    weight = torch.cat([model.classifier.fc.weight.data, tail_real])
    model.classifier.fc = nn.Linear(128, 54, bias=False)
    model.classifier.fc.weight.data = weight

    print("the time cost",data_time.val)
    print(format(data_time.val, '.8f'))
    print('imprint done')

def validate(val_loader, model):
    data_time = AverageMeter()
    microF1 = AverageMeter()
    test_p1, test_p3, test_p5 = 0, 0, 0
    test_ndcg1, test_ndcg3, test_ndcg5 = 0, 0, 0
    model.eval()
    with torch.no_grad():
        end = time.time()
        for batch_idx, (input, target) in enumerate(val_loader):
            data_time.update(time.time() - end)

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
            output[output  <= 0.5] = 0
            micro, macro = calc_f1(target, output)
            microF1.update(micro.item(), input.size(0))

        np.set_printoptions(formatter={'float': '{: 0.4}'.format})
        print('the result of micro: \n',microF1.avg)
        test_p1 /= len(val_loader)
        test_p3 /= len(val_loader)
        test_p5 /= len(val_loader)

        test_ndcg1 /= len(val_loader)
        test_ndcg3 /= len(val_loader)
        test_ndcg5 /= len(val_loader)

        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (test_p1, test_p3, test_p5))
        print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg1, test_ndcg3, test_ndcg5))
        return (microF1.avg)


def fine_tuning(train_loader, model, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    microF1 = AverageMeter()
    macroF1 = AverageMeter()
    model.train()

    end = time.time()
    bar = Bar('Training', max=len(train_loader))
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


        batch_time.update(time.time() - end)
        end = time.time()

        model.weight_norm()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Micro-f1: {microF1: .4f} |Macro-f1: {macroF1: .4f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            microF1=microF1.avg,
            macroF1=macroF1.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, microF1.avg, macroF1.avg)


if __name__ == '__main__':
    main()