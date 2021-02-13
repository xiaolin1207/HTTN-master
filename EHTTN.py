import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import torch.optim
import torch.utils.data
import models
import data_got
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=45, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-c', '--checkpoint', default='imprint_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: imprint_checkpoint)')
parser.add_argument('--model', default='samp45f_pretrain_checkpoint/model_best.pth.tar', type=str, metavar='PATH',
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
parser.add_argument('--epochs', default=7 ,type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--Bsamp_freq', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--Nsamp_freq', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--samp_freq', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate')


def main():
    global args, best_micro
    args = parser.parse_args()

    base_transf, embed = data_got.one_sample_base2avg(batch_size=args.batch_size,sample_num=args.num_sample,samp_freq=args.Bsamp_freq,num_class=args.num_class)
    Ftest_loader, novel_loader,novelall_loader,test_y = data_got.Nload_data(batch_size=args.batch_size,sample_num=args.num_sample,samp_freq=args.Nsamp_freq,num_class=args.num_class)
    embed = torch.from_numpy(embed).float()
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
    optimizer = torch.optim.Adam( trans_model.parameters(), lr=0.013, betas=(0.9, 0.99))

    for epoch in range(args.epochs):
        train_loss= train(base_transf, trans_model,model, criterion,  optimizer,real_weight)
        print("loss",train_loss)
    tail_weight=imprint(novel_loader, model,trans_model)
    # model_criterion= nn.BCELoss()
    #
    # model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # for i in range(3):
    #     train_loss, trn_micro, trn_macro = fine_tuning(novelall_loader, model,  model_criterion, model_optimizer)
    output_all=[]
    F1 = np.zeros(54)
    for i in range(args.Nsamp_freq):
        print("ensemble start!!!!!!!! this is calssifier",i)
        model.classifier.fc.weight.data = tail_weight[i]
        output = validate(Ftest_loader, model)
        output_all.append(output)
    output_all =(torch.sum(torch.tensor(output_all),0))/args.Nsamp_freq
    output_all[output_all > 0.5] = 1
    output_all[output_all<= 0.5] = 0
    for l in range(54):
        F1[l] = f1_score(test_y[:, l], output_all[:, l], average='binary')
    print("each class result f1!!!!!!!!!!!!!!")
    print(F1)

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
    base_rep= torch.from_numpy(base_rep).cuda()
    new_weight = torch.zeros(2700, 128).cuda()
    j = 0
    for i in range(2700):
        tmp =base_rep[j:j + args.num_sample]
        tmp = torch.sum(tmp, 0) / args.num_sample
        new_weight[i] = tmp / tmp.norm(p=2)
        j = j + args.num_sample
    e=0
    for h in range(args.Bsamp_freq):
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
    new_weight = torch.zeros(270, 128).cuda()
    j=0
    for i in range(270):
        tmp = output_stack[j:j + args.num_sample]
        tmp = torch.sum(tmp, 0) / args.num_sample
        new_weight[i] = tmp / tmp.norm(p=2)
        j = j + args.num_sample
    tail_real=trans_model.transfor(new_weight)
    tail_sampfre=[]
    j=0
    for i in range(args.Nsamp_freq):
        qq = tail_real[j:j + 9]
        j=j+9
        weight = torch.cat([model.classifier.fc.weight.data, qq])
        tail_sampfre.append(weight)
    print('imprint done')
    return tail_sampfre


def validate(val_loader, model):
    one_iteration=[]
    F1 = np.zeros(54)
    model.eval()

    with torch.no_grad():

        for batch_idx, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            target = target.data.cpu().float()
            output = output.data.cpu()
            one_iteration.extend(output.numpy())
            for l in range(54):
                F1[l] += f1_score(target[:, l], output[:, l], average='binary')
        np.set_printoptions(formatter={'float': '{: 0.4}'.format})
        print('the result of F1: \n', F1/len(val_loader))
        return one_iteration



def fine_tuning(train_loader, model, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    microF1 = AverageMeter()
    macroF1 = AverageMeter()
    model.train()

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
        #
        # micro,macro = calc_f1( target,  output)


        losses.update(loss.item(), input.size(0))
        # microF1.update(micro.item(), input.size(0))
        # macroF1.update(macro.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # model.weight_norm()
        # plot progress
    #     bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Micro-f1: {microF1: .4f} |Macro-f1: {macroF1: .4f}'.format(
    #         batch=batch_idx + 1,
    #         size=len(train_loader),
    #         data=data_time.val,
    #         bt=batch_time.val,
    #         total=bar.elapsed_td,
    #         eta=bar.eta_td,
    #         loss=losses.avg,
    #         microF1=microF1.avg,
    #         macroF1=macroF1.avg,
    #     )
    #     bar.next()
    # bar.finish()
    return (losses.avg, microF1.avg, macroF1.avg)


if __name__ == '__main__':
    main()