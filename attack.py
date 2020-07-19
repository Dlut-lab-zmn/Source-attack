from __future__ import print_function

import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
from torch.autograd import Variable
from advertorch.attacks import L2PGDAttack, L2BasicIterativeAttack, PGDAttack, MomentumIterativeAttack
from advertorch.attacks import LinfPGDAttack
from advertorch.attacks import JSMA
from advertorch.attacks import CarliniWagnerL2Attack
from keras.utils import to_categorical
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
from models import *
from utils import progress_bar
from models import *
from scipy.misc import imsave


# from dataload import load_file_list,load_test_list,get_batch,get_test
device = 'cuda'
# np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(
    description='PGD Adversarial Attacks using GuidedComplementEntropy')
parser.add_argument('--GCE', action='store_true',
                    help='Using GuidedComplementEntropy as a loss function for crafting adversarial examples')
parser.add_argument('--AVG', action='store_true',
                    help='Using AVG')
parser.add_argument('--alpha', '-a', default=0.333, type=float,
                    help='alpha for guiding factor')  # ckpt.GuidedComplementEntropy   ckpt.CrossEntropyLoss
parser.add_argument('--model',
                    default='',
                    type=str,
                    help='load a training model from your (physical) path')#
parser.add_argument('--attack_for_twoc',
                    default=False,
                    type=bool,
                    help='transfer the data stream')

parser.add_argument('--batch-size', '-b', default=64,
                    type=int, help='mini-batch size (default: 64)')
parser.add_argument('--eps', '-e', default=50., type=float,
                    help='Set an eplison value for PGD adversarial attacks')
parser.add_argument('--print-freq', '-p', default=40, type=int,
                    metavar='N', help='print frequency (default: 20)')
args = parser.parse_args()
net = couple_Net.__dict__['couple_Net']()


# net = VGG('VGG19')
# net2 = LeNet()
# net3 = ResNet18()
# net4 = c4('C4')
# net6 = LeNet()
# net2 = ResNet18()
# net = PreActResNet18()
# net5 = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()

class cross_entropy_loss(object):
    def __init__(self):
        super(cross_entropy_loss, self).__init__()

    # convert out to softmax probability
    def __call__(self, input, label):
        # prob = torch.clamp(torch.softmax(input, 1), 1e-10, 1.0)
        #prob = torch.clamp(1 * torch.softmax(input[:, :14], 1), 1e-10, 1.0)
        prob = torch.clamp(2 * torch.softmax(input[:,:19], 1), 1e-10, 2.0)
        #prob = torch.clamp(3 * torch.softmax(input, 1), 1e-10, 3.0)
        
        # loss = torch.sum(-label * torch.log(prob + 1e-8))
        # loss2 = torch.sum(-mlabel * torch.log(prob2 + 1e-8))
        loss = torch.sum(-label[:,:19]* torch.log(prob[:,:19] + 1e-8))
        #loss = torch.sum(-label* torch.log(prob + 1e-8))
        # loss = 0.2 * loss1 + 0.5 * loss2 + 0.3* loss3
        # cost4 = tf.reduce_sum(tf.abs(self.logits_scaled2[:, :14] - self.logits_scaled1[:, :14]))
        # cost5 = tf.reduce_sum(tf.abs(self.logits_scaled3[:, :19] - self.logits_scaled2[:, :19]))
        return loss


def accuracy(data1, data2, value):
    temp1 = MaxNum(data1, value)
    temp2 = MaxNum(data2, value)
    return np.mean(acc(temp1, temp2))


def MaxNum(nums, value):
    temp1 = []
    batch_size = nums.size()[0]
    nums = list(nums)
    for i in range(batch_size):
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


net.cuda()
# net2 = net2.to(device)
# net3 = net3.to(device)
# net4 = net4.to(device)
# net5 = net5.to(device)
# net6 = net6.to(device)


print('==> Resuming from checkpoint..')

# self.model.features = torch.nn.DataParallel(self.model.features)
checkpoint = torch.load(args.model)
net.load_state_dict(checkpoint['state_dict'])
csl = cross_entropy_loss()
print('==> Resuming down!')
cudnn.benchmark = True
if args.GCE:
    adversary = LinfPGDAttack(
        net, loss_fn=cross_entropy_loss(), eps=args.eps,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)
else:

        adversary = L2PGDAttack(net, loss_fn=csl, eps=args.eps, nb_iter=10, eps_iter=12.75,rand_init=True, clip_min=0.0, clip_max=255., targeted=False)
        #adversary = PGDAttack(net, loss_fn=csl, eps=args.eps, nb_iter=10, eps_iter=12.75, rand_init=True, clip_min=0.0,clip_max=255., targeted=False)
        #adversary = L2BasicIterativeAttack(net, loss_fn=csl, eps=args.eps, nb_iter=10, eps_iter=12.75, clip_min=0.0, clip_max=255., targeted=False)
        #adversary =MomentumIterativeAttack(net, loss_fn=csl, eps=args.eps, nb_iter=10, eps_iter=12.75, clip_min=0.0, clip_max=255., targeted=False)
        # adversary = JSMA(net, num_classes = 24, clip_min=0.0, clip_max=255.,theta=1.0, gamma=1.0,loss_fn=nn.CrossEntropyLoss())

net.eval()
# net2.eval()
# net3.eval()
# net4.eval()
# net5.eval()
correct = 0
total = 0
if args.attack_for_twoc:
    from dataload_test import load_test_list, load_test_list2, get_test, get_test2
    test_num = load_test_list()
    test_num2 = load_test_list2()
    iters2 = test_num // args.batch_size
    iters3 = test_num2 // args.batch_size
    if iters2 < iters3:
        iters = iters2
    else:
        iters = iters3
else:
    from dataload import load_test_list, get_test
    test_num = load_test_list()
    iters = test_num // args.batch_size
print(iters)
prec1 = AverageMeter()
prec2 = AverageMeter()
prec3 = AverageMeter()
prec4 = AverageMeter()
prec5 = AverageMeter()
noise_avg = 0
for iter in range(iters):

    inputt, labelt, mlabelt, blabelt = get_test(args.batch_size)
    inputt = torch.FloatTensor(inputt)
    labelt = torch.FloatTensor(labelt)
    mlabelt = torch.FloatTensor(mlabelt)
    blabelt = torch.FloatTensor(blabelt)
    inputt = inputt.cuda()
    labelt = labelt.cuda()
    mlabelt = mlabelt.cuda()
    blabelt = blabelt.cuda()
    """
    inputp, labelp, mlabelp, blabelp = get_test2(args.batch_size)
    inputp = torch.FloatTensor(inputp)
    labelp = torch.FloatTensor(labelp)
    mlabelp = torch.FloatTensor(mlabelp)
    blabelp = torch.FloatTensor(blabelp)
    labelp = labelp.cuda()
    mlabelp = mlabelp.cuda()
    blabelp = blabelp.cuda()
    inputp = inputp.cuda()
    """

    adv_inputs = adversary.perturb(inputt, labelt)  # inputs
    #adv_inputs = inputt

    noise = torch.normal(0,1,inputt.size()).cuda()
    adv_inputs = torch.clamp(adv_inputs+noise,0,255)
    # print(torch.mean(torch.abs())
    outt, outt2, outt3 = net(adv_inputs)
    out1,out2,out3 = net(inputt)
    noise_avg += torch.mean(torch.abs(adv_inputs - inputt))

    if iter % args.print_freq == 0:
        top1 = accuracy(outt.data, blabelt, 1)
        top2 = accuracy(outt2.data, mlabelt, 2)
        #top3 = accuracy(outt3.data, labelt, 3)
        top3 = accuracy(outt3.data[:,:19], labelt[:,:19], 2)
        top4 = accuracy(outt3.data[:, :19], outt2[:, :19], 2)
        top5 = accuracy(outt3.data[:, :14], outt[:, :14], 1)
        prec1.update(top1.item(), inputt.size(0))
        prec2.update(top2.item(), inputt.size(0))
        prec3.update(top3.item(), inputt.size(0))
        prec4.update(top4.item(), inputt.size(0))
        prec5.update(top5.item(), inputt.size(0))
        print('Epoch: [{0}][{1}/{2}]\t'
              'Prec@1 {top1:.3f}\t'
              'Prec@2 {top2:.3f}\t'
              'Prec@3 {top3:.3f}'.format(
            0, iter, iters,
            top1=top1, top2=top2, top3=top3
        ))
print(' * Prec@1 {top1.avg:.3f}'.format(top1=prec1))
print(' * Prec@1 {top2.avg:.3f}'.format(top2=prec2))
print(' * Prec@1 {top3.avg:.3f}'.format(top3=prec3))
print(' * Prec@1 {top4.avg:.3f}'.format(top4=prec4))
print(' * Prec@1 {top5.avg:.3f}'.format(top5=prec5))
print(' * noise_avg@1 {noise:.3f}'.format(noise=noise_avg / iters))