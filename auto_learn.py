#Author Lt Zhao
from models import *
import torch
import torch.nn as nn
arch = 'couple_Net'
resume = ""#load pre-trained couple_net model
__all__ = [
    'auto_learn',
]
class Auto_learn(nn.Module):
    def __init__(self,arch,resume,attribute,joint,joint2):
        super(Auto_learn, self).__init__()
        self.model = couple_Net.__dict__[arch]()
        #self.model.features = torch.nn.DataParallel(self.model.features)
        checkpoint = torch.load(resume)
        self.model.load_state_dict(checkpoint['state_dict'])
        for k,v in self.model.named_parameters():
                v.requires_grad=False
        self.attribute = attribute
        self.joint = joint
        self.joint2 = joint2
    def forward(self, xt,xp):
        att_t = self.attribute(xt)
        att_p = self.attribute(xp)
        fea = torch.cat((att_t,att_p),1)
        att_t = self.joint(fea)
        att_p = self.joint2(fea)
        att_t = torch.mean(att_t,0,True)
        att_p = torch.mean(att_p,0,True)
        #noise1 = (att_p- att_t)#.int().float()
        
        xt = torch.clamp(xt +att_t+128.,0,255)
        xp = torch.clamp(xp +att_p+128.,0,255)
        outt,outt2,outt3 = self.model(xt)
        outp,outp2,outp3 = self.model(xp)
        return outt,outt2,outt3,outp,outp2,outp3, att_t,att_p
def attribute(batch_norm=False):
    res = [64, 64, 64]
    layers = []
    in_channels = 3
    for v in res:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)
def joint(batch_norm=False):
    res = [128, 128, 3]
    layers = []
    in_channels = 128
    for v in res:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)
def joint2(batch_norm=False):
    res = [128, 128, 3]
    layers = []
    in_channels = 128
    for v in res:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)
def auto_learn():
    """VGG 11-layer model (configuration "A")"""
    return Auto_learn(arch,resume,attribute(),joint(),joint2())