
import argparse
import time
from dataload_auto_learn import load_file_list, load_file_list2, load_test_list, load_test_list2, \
    get_batch, get_test, get_batch2, get_test2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import process
import auto_learn
import os
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--repeat_epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to repeat')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=48, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=40, type=int,
                    metavar='N', help='print frequency (default: 20)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--class_net_path', dest='class_net_path',
                    help='The directory used to save the trained models',
                    default='Universal', type=str)


def one_hot(a, n):
    a = a.cpu()
    b = a.shape[0]
    c = np.zeros([b, n])
    for i in range(b):
        c[i][int(a[i])] = 1
    return c


def cross_entropy_loss(output, label):
    # convert out to softmax probability
    if len(label) > 1 and len(output)>1:
        #print("triple loss")
        out1, out2, out3 =  output
        blabel, mlabel, dlabel = label
        prob = torch.clamp(torch.softmax(out1, 1), 1e-10, 1.0)
        prob2 = torch.clamp(2 * torch.softmax(out2, 1), 1e-10, 2.0)
        prob3 = torch.clamp(3 * torch.softmax(out3, 1), 1e-10, 3.0)
        prob4 = torch.clamp(2*torch.softmax(out3[:,:19], 1), 1e-10, 2.0)
    
        loss1 = torch.sum(-blabel * torch.log(prob + 1e-8))
        loss2 = torch.sum(-mlabel * torch.log(prob2 + 1e-8))
        loss3 = torch.sum(-dlabel * torch.log(prob3 + 1e-8))
        loss4 = torch.sum(-dlabel[:,:19]*torch.log(prob4 + 1e-8))
        # loss = 0.2 * loss1 + 0.5 * loss2 + 0.3* loss3
        loss = loss3
    elif len(label) > 1 or len(output)>1:
        print("output not match")
        assert tuple == int
    else:
        prob = torch.clamp(torch.softmax(output, 1), 1e-10, 1.0)
    
        loss = torch.sum(-label * torch.log(prob + 1e-8))

    return loss

def to_float(output1,output2):
    if isinstance(output1, tuple) and isinstance(output2, tuple):
        outt,outt2,outt3 = output1
        outp,outp2,outp3 = output2
        outt = outt.float()
        outt2 = outt2.float()
        outt3 = outt3.float()
        outp = outp.float()
        outp2 = outp2.float()
        outp3 = outp3.float()
        return (outt,outt2,outt3),(outp,outp2,outp3)
    elif isinstance(output1, tuple) or isinstance(output2, tuple):
        print("output not match")
        assert tuple == int
    else:
        output1 = output1.float()
        output2 = output2.float()
        return output1,output2

def dataload_train():
    train_num = load_file_list()
    train_num2 = load_file_list2()
    return train_num,train_num2
def dataload_test():
    test_num = load_test_list()
    test_num2 = load_test_list2()
    return test_num,test_num2

def to_cuda(inputt, inputp, labeltt,labeltp):
    if isinstance(labeltt, tuple) and isinstance(labeltp, tuple):
        labelt, mlabelt, clabelt = labeltt
        labelp, mlabelp, clabelp = labeltp
        inputt = inputt.cuda(async=True)
        labelt = labelt.cuda(async=True)
        mlabelt = mlabelt.cuda(async=True)
        clabelt = clabelt.cuda(async=True)
        inputp = inputp.cuda(async=True)
        labelp = labelp.cuda(async=True)
        mlabelp = mlabelp.cuda(async=True)
        clabelp = clabelp.cuda(async=True)
        return (inputt, inputp), (labelt, mlabelt, clabelt), (labelp, mlabelp, clabelp)
    elif isinstance(labeltt, tuple) or isinstance(labeltp, tuple):
        print("output not match")
        assert tuple == int
    else:
        inputt = inputt.cuda(async=True)
        inputp = inputp.cuda(async=True)
        labeltt = labeltt.cuda(async=True)
        labeltp = labeltp.cuda(async=True)
        return inputt,inputp,labeltt,labeltp
def main():
    global args, best_prec1

    args = parser.parse_args()

    # Check the save_dir exists or not

    train_num,train_num2 = dataload_train()
    test_num,test_num2 = dataload_test()
    # define loss function (criterion) and pptimizer
    criterion = cross_entropy_loss  # nn.CrossEntropyLoss()


    args.arch = 'auto_learn'
    in_model = auto_learn
    save_path = args.save_dir_auto
    model_resume = os.path.join(save_path, 'checkpoint_{}.tar'.format(10))
    model_init = os.path.join(args.class_net_path, 'checkpoint_{}.tar'.format(20))

    best_prec1 = 0
    model = in_model.__dict__[args.arch](model_init)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # model.features = torch.nn.DataParallel(model.features)
    if args.cpu:
        model.cpu()
    else:
        model.cuda()

    # optionally resume from a checkpoint
    if model_resume:
        if os.path.isfile(model_resume):
            print("=> loading checkpoint '{}'".format(model_resume))
            checkpoint = torch.load(model_resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_resume))

    cudnn.benchmark = True


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.evaluate:
        validate(test_num, test_num2, model, criterion,repeat_epoch//2,flag = args.arch)
        return

    validate(train_num, train_num2, model, criterion,1,'train',flag = args.arch)
    validate(test_num, test_num2, model, criterion,1,'test',flag = args.arch)

def init_AverageMeter(num):
    top = []
    for i in range(num):
        top.append(AverageMeter())
    return top
def save_imgs(adv,save_path,dirs,iter):
    file_path = os.path.join(save_path,dirs)
    for i in range(adv.shape[0]):
        img_path = os.path.join(file_path,str(iter*adv.shape[0]+i)+'.png')
        adv_img = adv[i].reshape(64,64,3)
        adv_img = np.float32(adv_img)
        cv2.imwrite(img_path,adv_img)
    #img = cv2.imread(img_path)
    #print(img - np.array(adv_img))
    #print(img)
def validate(test_num, test_num2, model, criterion,epoch,dataset = 'train',flag = 'auto_learn'):
    """
    Run evaluation
    """
    assert flag in ['auto_learn','process']
    assert dataset in ['train','test']
    save_train_path = './S_64/train/adv/'
    save_test_path = './S_64/test/adv/'
    batch_time = AverageMeter()
    losses = AverageMeter()
    top = init_AverageMeter(10)
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
        if dataset == 'train':
            input1,label1 = get_batch(args.batch_size)
            input2,label2 = get_batch2(args.batch_size)
        else:
            input1,label1 = get_test(args.batch_size)
            input2,label2 = get_test2(args.batch_size)
        input1,input2 = torch.FloatTensor([input1,input2])
        label1,label2 = torch.FloatTensor([label1,label2])
        if args.cpu == False:
            input1,input2,label1,label2 = to_cuda(input1,input2,label1,label2)
        if args.half:
            inputt = inputt.half()
            inputp = inputp.half()

        # compute output
        with torch.no_grad():
            adv_t,adv_p,noise_t, noise_p = model(input1,input2)#outt, outt2, outt3, outp, outp2, outp3,
            if dataset == 'train':
                save_imgs(adv_t.cpu(),save_train_path,'train1',iter)
                save_imgs(adv_p.cpu(),save_train_path,'train10',iter)
            else:
                save_imgs(input1.cpu()+128.,save_test_path,'test1',iter)
                save_imgs(input2.cpu()+128.,save_test_path,'test10',iter)
            output1 = model.couple_net(adv_t)
            output2 = model.couple_net(adv_p)
            if flag == 'auto_learn':
                losst2p = criterion(output1, label2)
                lossp2t = criterion(output2, label1)
            else:
                losst2p = criterion(output1, label1)
                lossp2t = criterion(output2, label2)
            loss = losst2p + lossp2t

        output1,output2 = to_float(output1,output2)

        loss = loss.float()
        # measure accuracy and record loss

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iter % args.print_freq == 0:
            # Accuracy t->p  top1~top3
            # Accuracy p->t  top4~top6
            if flag != 'auto_learn':
                  label3 = label1
                  label1 = label2
                  label2 = label3
            prec0 = accuracy(output2[0].data, label1[0], 1)
            top[0].update(prec0.item(), input1.size(0))
            
            prec1 = accuracy(output2[1].data, label1[1], 2)
            top[1].update(prec1.item(), input1.size(0))
            prec2 = accuracy(output2[2].data, label1[2], 3)
            top[2].update(prec2.item(), input1.size(0))
            
            prec3 = accuracy(output1[0].data, label2[0], 1)
            top[3].update(prec3.item(), input1.size(0))
            prec4 = accuracy(output1[1].data, label2[1], 2)
            top[4].update(prec4.item(), input1.size(0))
            prec5 = accuracy(output1[2].data, label2[2], 3)
            top[5].update(prec5.item(), input1.size(0))

            # Accuracy relation match degree  top7~top8
            prec6 = accuracy(output2[2].data[:, :19], output2[1].data[:, :19], 2)
            prec7 = accuracy(output2[2].data[:, :14], output2[0].data[:, :14], 1)
            top[6].update(prec6.item(), input1.size(0))
            top[7].update(prec7.item(), input1.size(0))
            
            prec8 = accuracy(output1[2].data[:, :19], output1[1].data[:, :19], 2)
            prec9 = accuracy(output1[2].data[:, :14], output1[0].data[:, :14], 1)
            top[8].update(prec8.item(), input1.size(0))
            top[9].update(prec9.item(), input1.size(0))
            
            losses.update(loss.item(), input1.size(0))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@0 {top0:.3f}\t'
                  'Prec@1 {top1:.3f}\t'
                  'Prec@2 {top2:.3f}\t'
                  'Prec@3 {top3:.3f}\n'
                  'Prec@4 {top4:.3f}\t'
                  'Prec@5 {top5:.3f}\t'
                  'Prec@6 {top6:.3f}\t'
                  'Prec@7 {top7:.3f}\t'
                  'Prec@8 {top8:.3f}\t'
                  'Prec@9 {top9:.3f}\t'.format(
                epoch, iter, iters, loss=losses, top0=prec0,top1=prec1, top2=prec2, top3=prec3,
                top4=prec4, top5=prec5, top6=prec6, top7=prec7, top8=prec8, top9=prec9
            ))




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
