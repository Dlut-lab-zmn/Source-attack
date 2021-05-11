import argparse
import time
from dataload_auto_learn import load_file_list, load_file_list2, load_test_list, load_test_list2, \
    get_batch, get_test, get_batch2, get_test2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import auto_learn
import os
import numpy as np

model_names = sorted(name for name in auto_learn.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("auto_learn")
                     and callable(auto_learn.__dict__[name]))
print(model_names)
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg',  # 'vgg19',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: vgg19)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
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


def cross_entropy_loss(out1, out2, out3, label, mlabel, clabel):
    # convert out to softmax probability

    prob = torch.clamp(torch.softmax(out1, 1), 1e-10, 1.0)
    prob2 = torch.clamp(2 * torch.softmax(out2, 1), 1e-10, 2.0)
    prob3 = torch.clamp(3 * torch.softmax(out3, 1), 1e-10, 3.0)

    loss1 = torch.sum(-clabel * torch.log(prob + 1e-8))
    loss2 = torch.sum(-mlabel * torch.log(prob2 + 1e-8))
    loss3 = torch.sum(-label * torch.log(prob3 + 1e-8))

    # loss = 0.2 * loss1 + 0.5 * loss2 + 0.3* loss3
    loss = loss3

    # cost4 = tf.reduce_sum(tf.abs(self.logits_scaled2[:, :14] - self.logits_scaled1[:, :14]))
    # cost5 = tf.reduce_sum(tf.abs(self.logits_scaled3[:, :19] - self.logits_scaled2[:, :19]))
    return loss


def main():
    global args, best_prec1

    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = auto_learn.__dict__[args.arch]()

    # model.features = torch.nn.DataParallel(model.features)
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
    train_num2 = load_file_list2()
    test_num2 = load_test_list2()
    # define loss function (criterion) and pptimizer
    criterion = cross_entropy_loss  # nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.evaluate:
        validate(test_num, model, criterion)
        return
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_num, train_num2, model, criterion, optimizer, epoch)
        # evaluate on validation set
        prec1 = validate(test_num, test_num2, model, criterion)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))


def train(train_num, train_num2, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()
    top4 = AverageMeter()
    top5 = AverageMeter()
    top6 = AverageMeter()

    top7 = AverageMeter()
    top8 = AverageMeter()
    top9 = AverageMeter()
    top10 = AverageMeter()
    sum_value = 0
    # switch to train mode
    model.train()
    end = time.time()
    iters2 = train_num // args.batch_size
    iters3 = train_num2 // args.batch_size
    if iters2 < iters3:
        iters = iters2
    else:
        iters = iters3
    for iter in range(iters):
        # measure data loading time
        data_time.update(time.time() - end)
        inputt, labelt, mlabelt, clabelt = get_batch(args.batch_size)
        inputp, labelp, mlabelp, clabelp = get_batch2(args.batch_size)
        inputt = torch.FloatTensor(inputt)
        inputp = torch.FloatTensor(inputp)
        labelt = torch.FloatTensor(labelt)
        labelp = torch.FloatTensor(labelp)
        mlabelt = torch.FloatTensor(mlabelt)
        mlabelp = torch.FloatTensor(mlabelp)
        clabelt = torch.FloatTensor(clabelt)
        clabelp = torch.FloatTensor(clabelp)
        if args.cpu == False:
            inputt = inputt.cuda(async=True)
            labelt = labelt.cuda(async=True)
            mlabelt = mlabelt.cuda(async=True)
            clabelt = clabelt.cuda(async=True)
            inputp = inputp.cuda(async=True)
            labelp = labelp.cuda(async=True)
            mlabelp = mlabelp.cuda(async=True)
            clabelp = clabelp.cuda(async=True)
        if args.half:
            inputt = inputt.half()
            inputp = inputp.half()

        # compute output
        outt, outt2, outt3, outp, outp2, outp3, att_t, att_p = model(inputt, inputp)
        noise = torch.mean(abs(att_t - att_p))
        att_t = torch.mean(abs(att_t))
        att_p = torch.mean(abs(att_p))
        # noise1 = (att_p- att_t)
        # noi = torch.mean(torch.clamp(inputt +noise1+128.,0,255)-inputt);print(noi)
        loss1 = criterion(outt, outt2, outt3, labelp, mlabelp, clabelp)
        loss2 = criterion(outp, outp2, outp3, labelt, mlabelt, clabelt)
        loss = loss1 + loss2 + 0.4 * (att_t + att_p)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outt = outt.float()
        outt2 = outt2.float()
        outt3 = outt3.float()
        outp = outp.float()
        outp2 = outp2.float()
        outp3 = outp3.float()
        loss = loss.float()
        # measure accuracy and record loss

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        sum_value += noise
        if iter % args.print_freq == 0:
            # Accuracy t->p  top1~top3
            # Accuracy p->t  top4~top6
            prec1 = accuracy(outp.data, clabelt, 1)
            top1.update(prec1.item(), inputt.size(0))
            prec2 = accuracy(outp2.data, mlabelt, 2)
            top2.update(prec2.item(), inputt.size(0))
            prec3 = accuracy(outp3.data, labelt, 3)
            top3.update(prec3.item(), inputt.size(0))
            prec4 = accuracy(outt.data, clabelp, 1)
            top4.update(prec4.item(), inputt.size(0))
            prec5 = accuracy(outt2.data, mlabelp, 2)
            top5.update(prec5.item(), inputt.size(0))
            prec6 = accuracy(outt3.data, labelp, 3)
            top6.update(prec6.item(), inputt.size(0))
            losses.update(loss.item(), inputt.size(0))
            # Accuracy relation match degree  top7~top8
            prec7 = accuracy(outp3.data[:, :19], outp2[:, :19], 2)
            prec8 = accuracy(outp3.data[:, :14], outp[:, :14], 1)
            top7.update(prec7.item(), inputt.size(0))
            top8.update(prec8.item(), inputt.size(0))
            prec9 = accuracy(outt3.data[:, :19], outt2[:, :19], 2)
            prec10 = accuracy(outt3.data[:, :14], outt[:, :14], 1)
            top9.update(prec9.item(), inputt.size(0))
            top10.update(prec10.item(), inputt.size(0))

            losses.update(loss.item(), inputt.size(0))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1:.3f}\t'
                  'Prec@2 {top2:.3f}\t'
                  'Prec@3 {top3:.3f}\n'
                  'Prec@4 {top4:.3f}\t'
                  'Prec@5 {top5:.3f}\t'
                  'Prec@6 {top6:.3f}\t'
                  'Prec@7 {top7:.3f}\t'
                  'Prec@8 {top8:.3f}\t'
                  'Prec@9 {top9:.3f}\t'
                  'Prec@10 {top10:.3f}'.format(
                epoch, iter, iters, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=prec1, top2=prec2, top3=prec3,
                top4=prec4, top5=prec5, top6=prec6, top7=prec7, top8=prec8, top9=prec9, top10=prec10
            ))


def validate(test_num, test_num2, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()
    top4 = AverageMeter()
    top5 = AverageMeter()
    top6 = AverageMeter()

    top7 = AverageMeter()
    top8 = AverageMeter()
    top9 = AverageMeter()
    top10 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    iters2 = test_num // args.batch_size
    iters3 = test_num2 // args.batch_size
    if iters2 < iters3:
        iters = iters2
    else:
        iters = iters3
    for iter in range(iters):

        inputt, labelt, mlabelt, clabelt = get_test(args.batch_size)
        inputp, labelp, mlabelp, clabelp = get_test2(args.batch_size)
        inputt = torch.FloatTensor(inputt)
        inputp = torch.FloatTensor(inputp)
        labelt = torch.FloatTensor(labelt)
        labelp = torch.FloatTensor(labelp)
        mlabelt = torch.FloatTensor(mlabelt)
        mlabelp = torch.FloatTensor(mlabelp)
        clabelt = torch.FloatTensor(clabelt)
        clabelp = torch.FloatTensor(clabelp)
        if args.cpu == False:
            inputt = inputt.cuda(async=True)
            labelt = labelt.cuda(async=True)
            mlabelt = mlabelt.cuda(async=True)
            clabelt = clabelt.cuda(async=True)
            inputp = inputp.cuda(async=True)
            labelp = labelp.cuda(async=True)
            mlabelp = mlabelp.cuda(async=True)
            clabelp = clabelp.cuda(async=True)
        if args.half:
            inputt = inputt.half()
            inputp = inputp.half()

        # compute output
        with torch.no_grad():
            outt, outt2, outt3, outp, outp2, outp3, att_t, att_p = model(inputt, inputp)
            loss = criterion(outt, outt2, outt3, labelp, mlabelp, clabelp)
            loss2 = criterion(outp, outp2, outp3, labelt, mlabelt, clabelt)
            loss = loss + loss2

        outt = outt.float()
        outt2 = outt2.float()
        outt3 = outt3.float()
        outp = outp.float()
        outp2 = outp2.float()
        outp3 = outp3.float()
        loss = loss.float()
        # measure accuracy and record loss

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iter % args.print_freq == 0:
            prec1 = accuracy(outp.data, clabelt, 1)
            top1.update(prec1.item(), inputt.size(0))
            prec2 = accuracy(outp2.data, mlabelt, 2)
            top2.update(prec2.item(), inputt.size(0))
            prec3 = accuracy(outp3.data, labelt, 3)
            top3.update(prec3.item(), inputt.size(0))
            prec4 = accuracy(outt.data, clabelp, 1)
            top4.update(prec4.item(), inputt.size(0))
            prec5 = accuracy(outt2.data, mlabelp, 2)
            top5.update(prec5.item(), inputt.size(0))
            prec6 = accuracy(outt3.data, labelp, 3)
            top6.update(prec6.item(), inputt.size(0))
            losses.update(loss.item(), inputt.size(0))

            prec7 = accuracy(outp3.data[:, :19], outp2[:, :19], 2)
            prec8 = accuracy(outp3.data[:, :14], outp[:, :14], 1)
            top7.update(prec7.item(), inputt.size(0))
            top8.update(prec8.item(), inputt.size(0))
            prec9 = accuracy(outt3.data[:, :19], outt2[:, :19], 2)
            prec10 = accuracy(outt3.data[:, :14], outt[:, :14], 1)
            top9.update(prec9.item(), inputt.size(0))
            top10.update(prec10.item(), inputt.size(0))
            print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1:.3f}\t'
                  'Prec@2 {top2:.3f}\t'
                  'Prec@3 {top3:.3f}\n'
                  'Prec@4 {top4:.3f}\t'
                  'Prec@5 {top5:.3f}\t'
                  'Prec@6 {top6:.3f}'.format(
                iter, iters, batch_time=batch_time,
                loss=losses, top1=prec1, top2=prec2, top3=prec3,
                top4=prec4, top5=prec5, top6=prec6
            ))
    # print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # print(' * Prec@1 {top2.avg:.3f}'.format(top2=top2))
    # print(' * Prec@1 {top3.avg:.3f}'.format(top3=top3))
    # print(' * Prec@1 {top4.avg:.3f}'.format(top4=top4))
    # print(' * Prec@1 {top5.avg:.3f}'.format(top5=top5))
    # print(' * Prec@1 {top6.avg:.3f}'.format(top6=top6))
    # print(' * Prec@1 {top7.avg:.3f}'.format(top7=top7))
    # print(' * Prec@1 {top8.avg:.3f}'.format(top8=top8))
    # print(' * Prec@1 {top9.avg:.3f}'.format(top9=top9))
    # print(' * Prec@1 {top10.avg:.3f}'.format(top10=top10))


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


def accuracy(data1, data2, value):
    temp1 = MaxNum(data1, value)
    temp2 = MaxNum(data2, value)
    return np.mean(acc(temp1, temp2))


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
