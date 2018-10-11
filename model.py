from torch import nn


class Flatten(nn.Module):
    """
    Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """
    def forward(self, x):
        out = x.view(len(x), -1)
        return out

def test_module(nclasses):
    net = []

    net.append(nn.Dropout(0.2))
    net.append(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=(4, 5), padding=3, dilation=1, groups=1))
    net.append(nn.ReLU())
    net.append(nn.AvgPool2d((1, 6)))
    net.append(Flatten())
    net.append(nn.Linear(16, 100))
    net.append(nn.Linear(100, nclasses))
    net = nn.Sequential(*net)
    return net

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

    net.append(nn.Linear(1440, 1440 * 2))
    net.append(nn.Linear(1440 * 2, 192 * 2))
    net.append(nn.Linear(192 * 2, 100))
    net.append(nn.Linear(100, nclasses))

    net = nn.Sequential(*net)
    return net


if __name__=='__main__':
    import torchsummary

    net = all_cnn_module(127)
    print(net)
    print(torchsummary.summary(net, (1, 64, 9984)))
    # print(torchsummary.summary(net, (1, 64, 50)))
