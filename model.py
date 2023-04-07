import torch
from torch import nn


# 搭建神经网络，也考察你看文献的
class Mynet(nn.Module):
    def __init__(self):# 这里的初始化再次强调，别tm打成int
        super(Mynet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 规范来说，可以在这里测试网络的正确性
if __name__ == "__main__":
    my_net = Mynet()
    iinput = torch.ones((64, 3, 32, 32))
    ooutput = my_net(iinput)
    print(ooutput.shape)
    #Input = torch.ones((64, 3, 32, 32))
    #output = my_net(Input)
    #print(output.shape)









