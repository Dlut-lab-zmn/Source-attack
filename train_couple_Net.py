import argparse
import time
from dataload import load_file_list,load_test_list,get_batch,get_test
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import os
import numpy as np

from models import *
model_names = sorted(name for name in couple_Net.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("couple_Net")
                     and callable(couple_Net.__dict__[name]))
print(model_names)
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='couple_Net',  # 'vgg19',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: couple_Net)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=40, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='', type=str)

best_prec1 = 0


def one_hot(a, n):
    a = a.cpu()
    b = a.shape[0]
    c = np.zeros([b, n])
    for i in range(b):
        c[i][int(a[i])] = 1
    return c


def cross_entropy_loss(out1,out2,out3,label,mlabel,clabel):
    # convert out to softmax probability

    prob = torch.clamp(torch.softmax(out1, 1), 1e-10, 1.0)
    prob2 = torch.clamp(2*torch.softmax(out2, 1), 1e-10, 2.0)
    prob3 = torch.clamp(3*torch.softmax(out3, 1), 1e-10, 3.0)

    loss1 = torch.sum(-clabel * torch.log(prob + 1e-8))
    loss2 = torch.sum(-mlabel * torch.log(prob2 + 1e-8))
    loss3 = torch.sum(-label * torch.log(prob3 + 1e-8))

    loss = 0.2 * loss1 + 0.5 * loss2 + 0.3* loss3

    # cost4 = tf.reduce_sum(tf.abs(self.logits_scaled2[:, :14] - self.logits_scaled1[:, :14]))
    # cost5 = tf.reduce_sum(tf.abs(self.logits_scaled3[:, :19] - self.logits_scaled2[:, :19]))
    return loss


"""
--------------------------------------------------------------------------------
"""


def main():
    global args, best_prec1

    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = couple_Net.__dict__[args.arch]()

    #model.features = torch.nn.DataParallel(model.features)
    if args.cpu:
        model.cpu()
    else:
        model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_num = load_file_list()
    test_num = load_test_list()
    # define loss function (criterion) and pptimizer
    criterion = cross_entropy_loss  # nn.CrossEntropyLoss()
    # if args.cpu:
    #    criterion = criterion.cpu()
    # else:
    #    criterion = criterion.cuda()


    #optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.evaluate:
        validate(test_num, model, criterion)
        return
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)


        # train for one epoch

        train(train_num, model, criterion, optimizer, epoch)
        # evaluate on validation set
        prec1 = validate(test_num, model, criterion)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))



def train(train_num, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    iters = train_num//args.batch_size
    for iter in range(iters):
        # measure data loading time
        data_time.update(time.time() - end)
        input,label,mlabel,clabel = get_batch(args.batch_size)
        input = torch.FloatTensor(input)
        label = torch.FloatTensor(label)
        mlabel = torch.FloatTensor(mlabel)
        clabel = torch.FloatTensor(clabel)
        if args.cpu == False:
            input = input.cuda(async=True)
            label = label.cuda(async=True)
            mlabel = mlabel.cuda(async=True)
            clabel = clabel.cuda(async=True)
        if args.half:
            input = input.half()

        # compute output conB_Fea,conM_Fea,conD_Fea  clabel,mlabel,label
        out,out2,out3 = model(input)
        loss = criterion(out,out2,out3, label,mlabel,clabel)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        out = out.float()
        out2 = out2.float()
        out3 = out3.float()
        loss = loss.float()
        # measure accuracy and record loss

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iter % args.print_freq == 0:
            prec1 = accuracy(out.data, clabel,1)
            prec2 = accuracy(out2.data, mlabel,2)
            prec3 = accuracy(out3.data, label,3)
            losses.update(loss.item(), input.size(0))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1:.3f}\t'
                  'Prec@2 {top2:.3f}\t'
                  'Prec@3 {top3:.3f}'.format(
                epoch, iter, iters, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=prec1, top2=prec2, top3=prec3))



def validate(test_num, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()


    # switch to evaluate mode
    model.eval()


    iters = test_num//args.batch_size
    for iter in range(iters):
        # measure data loading time
        input,label,mlabel,clabel = get_test(args.batch_size)
        input = torch.FloatTensor(input)
        label = torch.FloatTensor(label)
        mlabel = torch.FloatTensor(mlabel)
        clabel = torch.FloatTensor(clabel)
        if args.cpu == False:
            input = input.cuda(async=True)
            label = label.cuda(async=True)
            mlabel = mlabel.cuda(async=True)
            clabel = clabel.cuda(async=True)
        if args.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            out,out2,out3 = model(input)
            loss = criterion(out,out2,out3, label,mlabel,clabel)

        out = out.float()
        out2 = out2.float()
        out3 = out3.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = np.array(accuracy(out.data, clabel,1))
        prec2 = np.array(accuracy(out2.data, mlabel,2))
        prec3 = np.array(accuracy(out3.data, label,3))
        losses.update(loss.item(), input.size(0))

        if iter % args.print_freq == 0:
            print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1:.3f}\t'
                  'Prec@2 {top2:.3f}\t'
                  'Prec@3 {top3:.3f}'.format(
                 iter, iters, batch_time=batch_time,
                 loss=losses, top1=prec1, top2=prec2, top3=prec3))

    # with open("record.txt", "a+") as f:
    #     f.write("\t" + 'blabel:'+'{top1:.3f}'.format(top1=prec1))
    #     f.write("\t" + 'mlabel:'+'{top1:.3f}'.format(top1=prec2))
    #     f.write("\t" + 'blabel:'+'{top1:.3f}'.format(top1=prec3))
    return prec1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(data1,data2,value):
    temp1 = MaxNum(data1, value)
    temp2 = MaxNum(data2, value)
    return np.mean(acc(temp1,temp2))
def MaxNum(nums, value):
    temp1 = []
    nums = list(nums)
    for i in range(args.batch_size):
        temp = []
        Inf = 0
        nt = list(nums[i])
        for t in range(value):
            temp.append(nt.index(max(nt)))
            nt[nt.index(max(nt))] = Inf
        temp.sort()
        temp1.append(temp)
    return temp1

def acc(temp, index):
    accuracy = []  # print(np.array(temp).shape)
    for k in range(len(temp)):
        accuracy.append((temp[k] == index[k]))
    return accuracy


if __name__ == '__main__':
    main()
