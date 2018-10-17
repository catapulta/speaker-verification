from torch import nn
from torch.nn import functional as F
import net_sphere
import torchvision
import numpy as np
import torch


class Flatten(nn.Module):
    """
    Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """
    def forward(self, x):
        out = x.view(len(x), -1)
        return out


class Tester(nn.Module):
    def __init__(self, nclasses):
        super(Tester, self).__init__()
        self.nclasses = nclasses
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=(1, 3), padding=0, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.rel1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=(1, 2), padding=0, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.rel2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=(1, 2), padding=0, dilation=1, bias=False)
        self.rel3 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=(1, 1), padding=0, dilation=1, bias=False)
        self.rel4 = nn.ReLU()
        self.pool1 = nn.AvgPool2d((1, 71))
        self.pool2 = nn.AvgPool2d((7, 1))
        self.flatten = Flatten()
        self.drop2 = nn.Dropout(0.2)
        self.lin1 = nn.Linear(2304, 512, bias=False)
        self.al = net_sphere.AngleLinear(512, self.nclasses)

    def forward(self, x):
        x = self.rel1(self.bn1(self.conv1(x)))
        x = self.rel2(self.bn2(self.conv2(x)))
        x = self.rel3(self.conv3(x))
        x = self.rel4(self.conv4(self.drop1(x)))
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.drop2(x)
        x = self.lin1(x)
        x = self.al(x)
        return x

def all_cnn_module(nclasses):
    net = []

    net.append(nn.Dropout(0.2))
    net.append(nn.Conv2d(in_channels=1, out_channels=96, kernel_size=7, stride=(1, 4), padding=0, dilation=1, groups=1))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=5, stride=(1, 2), padding=0, dilation=1, groups=1))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=5, stride=(1, 2), padding=0, dilation=1, groups=1))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=(1, 2), padding=0, dilation=1, groups=1))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=(1, 2), padding=0, dilation=1, groups=1))
    net.append(nn.ReLU())

    net.append(nn.Dropout(0.5))
    net.append(nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=0))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=0))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=(1, 2), padding=0))
    net.append(nn.ReLU())

    net.append(nn.Dropout(0.5))
    net.append(nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=192, out_channels=20, kernel_size=1, stride=1, padding=0))
    net.append(nn.ReLU())

    net.append(nn.AvgPool2d((1, 8)))
    net.append(nn.AvgPool2d((5, 1)))
    net.append(Flatten())

    # net.append(nn.Linear(1440, 1440 * 2))
    # net.append(nn.Linear(1440 * 2, 192 * 2))
    # net.append(nn.Linear(192 * 2, 100))
    # net.append(nn.Linear(100, nclasses))

    net = nn.Sequential(*net)
    return net


class AudioDenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, classnum):
        super(AudioDenseNet121, self).__init__()
        self.strider = nn.Conv2d(1, 3, (7, 15), (1, 8), (3, 0))
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 29))
        self.embeddings = nn.Linear(4096, 300, bias=False)
        self.al = net_sphere.AngleLinear(300, classnum)
        self.alpha = torch.from_numpy(np.array(16)).float().cuda()

    def forward(self, x):
        x = self.strider(x)
        x = F.elu(x)
        x = self.densenet121.features(x) # use only features
        x = F.relu(x, inplace=True)
        x = self.avg_pool(x).view(x.size(0), -1)
        x = self.embeddings(x)
        x = F.normalize(x) * self.alpha
        x = self.al(x)
        return x


if __name__=='__main__':
    import torchsummary

    # net = all_cnn_module(127)
    # print(torchsummary.summary(net, (1, 64, 384)))

    # net = Tester(127)
    # print(torchsummary.summary(net, (1, 64, 5184)))
    # print(net._modules)

    net = AudioDenseNet121(127)
    print(torchsummary.summary(net, (1, 64, 468*32)))
    # print(net._modules)

    # net = net_sphere.sphere20a(127)
    # print(torchsummary.summary(net, (1, 64, 384)))
