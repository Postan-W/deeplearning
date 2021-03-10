import torch
from torchsummary import summary
import numpy as np
# # 下面是加载完整模型结构
# model = torch.load("./models/densenet169-b2777c0a.pth",map_location=torch.device('cpu'))
# # print(torch.cuda.is_available())
# # print(model)
#
# datafile = open("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\data.pkl","rb")
# labelsfile = open("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\labels.pkl","rb")
# import pickle
# data = pickle.load(datafile)
# #data归一化一下
# data = np.array(data,dtype="float32")
# data /= 255.0
"""
我们经常会看到后缀名为.pt, .pth, .pkl的pytorch模型文件，这几种模型文件在格式上有什么区别吗？

其实它们并不是在格式上有区别，只是后缀不同而已（仅此而已），在用torch.save()函数保存模型文件时，各人有不同的喜好，有些人喜欢用.pt后缀，有些人喜欢用.pth或.pkl.用相同的torch.save（）语句保存出来的模型文件没有什么不同。

在pytorch官方的文档/代码里，有用.pt的，也有用.pth的。一般惯例是使用.pth,但是官方文档里貌似.pt更多，而且官方也不是很在意固定用一种。
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_c, out_c, ngf=64):
        super(Generator, self).__init__()
        model = []
        model += [
            nn.Conv2d(in_c, ngf, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(ngf),
            nn.Conv2d(ngf, out_c, 3, 2, 1)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def save_entiremodel():
    netG = Generator(3, 3)
    input = torch.zeros(10, 3, 256, 256)
    output = netG(input)
    print("保存前的输出：", output)
    torch.save(netG, './models/netG.pth')


def load_entiremodel(path):
    netC = torch.load(path)
    # input = torch.zeros(10, 3, 256, 256)
    # output = netC(input)
    # print("保存后的输出:", output)
    try:
        print(netC.state_dict())
    except Exception as e:
        print(e)
    print(netC)


def save_parameter_only():
    netG = Generator(3, 3)
    torch.save({'netG': netG.state_dict()}, './models/netG_parameter.pth')


def load_parameter():
    netG = Generator(3, 3)
    state_dict = torch.load('./models/netG_parameter.pth')
    netG.load_state_dict(state_dict['netG'])#所以这种方式可以保存多个模型参数，用key区分
    input = torch.zeros(10, 3, 256, 256)
    output = netG(input)
    print(output)

load_entiremodel('./models/resnet18-5c106cde.pth')
