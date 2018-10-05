from torch import nn


class Flatten(nn.Module):
    """
    Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """
    def forward(self, x):
        out = x.view(len(x), -1)
        return out


def all_cnn_module(nclasses):
    net = []

    net.append(nn.Dropout(0.2))
    net.append(nn.Conv1d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, dilation=1, groups=1))
    net.append(nn.ReLU())
    net.append(nn.Conv1d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, dilation=1, groups=1))
    net.append(nn.ReLU())
    net.append(nn.Conv1d(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=1, dilation=1, groups=1))
    net.append(nn.ReLU())

    net.append(nn.Dropout(0.5))
    net.append(nn.Conv1d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1))
    net.append(nn.ReLU())
    net.append(nn.Conv1d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1))
    net.append(nn.ReLU())
    net.append(nn.Conv1d(in_channels=384, out_channels=384, kernel_size=3, stride=2, padding=1))
    net.append(nn.ReLU())

    net.append(nn.Dropout(0.5))
    net.append(nn.Conv1d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1))
    net.append(nn.ReLU())
    net.append(nn.Conv1d(in_channels=384, out_channels=384, kernel_size=1, stride=1, padding=0))
    net.append(nn.ReLU())
    net.append(nn.Conv1d(in_channels=384, out_channels=192, kernel_size=1, stride=1, padding=0))
    net.append(nn.ReLU())

    net.append(nn.AvgPool1d(5))
    net.append(Flatten())

    net.append(nn.Linear(192 * 500, 192 * 16))
    net.append(nn.Linear(192 * 16, 100))
    net.append(nn.Linear(100, nclasses))

    net = nn.Sequential(*net)
    return net


if __name__=='__main__':
    import torchsummary

    net = all_cnn_module(127)
    print(net)
    print(torchsummary.summary(net, (64, 10000)))
