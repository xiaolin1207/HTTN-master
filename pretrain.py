import argparse
import os
import shutil
import time
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data

import models
import numpy as np
import data_got

from utils import Bar, Logger, AverageMeter,  precision_k,  ndcg_k,calc_f1, mkdir_p, savefig
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=90, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-c', '--checkpoint', default='samp36_pretrain_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: pretrain_checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--lstm_hid_dim', default=150, type=int, metavar='N',
                    help='lstm_hid_dim')
parser.add_argument('--num_class', default=36, type=int, metavar='N',
                    help='the number of class')
parser.add_argument('--num-sample', default=5, type=int,
                    metavar='N', help='number of novel sample (default: 1)')
best_micro = 0

torch.cuda.set_device(1)
def main():
    global args, best_micro
    args = parser.parse_args()
    train_loader, val_loader, embed = data_got.Bload_data(batch_size=args.batch_size,num_class=args.num_class)
    embed = torch.from_numpy(embed).float()
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    model = models.Net(embed,args.lstm_hid_dim,args.num_class).cuda()

    criterion = nn.BCELoss()
    extractor_params = list(map(id, model.extractor.parameters()))
    classifier_params = filter(lambda p: id(p) not in extractor_params, model.parameters())
    optimizer = torch.optim.Adam([
                {'params': model.extractor.parameters()},
                {'params': classifier_params, 'lr': args.lr * 10}
            ], lr=args.lr,  betas=(0.9, 0.99))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    title = 'AAPD'
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_micro = checkpoint['best_micro']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        lr = optimizer.param_groups[1]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        train_loss= train(train_loader, model, criterion, optimizer, epoch)

        micro = validate(val_loader, model, criterion)
        val_micro=micro[2]

        is_best =val_micro > best_micro
        best_micro = max(val_micro, best_micro)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'val_micro': best_micro,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    savefig(os.path.join(args.checkpoint, 'log.eps'))
    print('Best val_micro:')
    print(val_micro)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    model.train()

    for batch_idx, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target.float())
        losses.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    score_micro = np.zeros(3)
    test_p1, test_p3, test_p5 = 0, 0, 0
    test_ndcg1, test_ndcg3, test_ndcg5 = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(val_loader):

            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target.float())
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
            score_micro += [precision_score(target, output, average='micro'),
                            recall_score(target, output, average='micro'),
                            f1_score(target, output, average='micro')]

            losses.update(loss.item(), input.size(0))

            # plot progress
        np.set_printoptions(formatter={'float': '{: 0.4}'.format})
        print('the result of micro: \n', score_micro / len(val_loader))
        test_p1 /= len(val_loader)
        test_p3 /= len(val_loader)
        test_p5 /= len(val_loader)

        test_ndcg1 /= len(val_loader)
        test_ndcg3 /= len(val_loader)
        test_ndcg5 /= len(val_loader)

        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (test_p1, test_p3, test_p5))
        print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg1, test_ndcg3, test_ndcg5))

        return score_micro / len(val_loader)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()