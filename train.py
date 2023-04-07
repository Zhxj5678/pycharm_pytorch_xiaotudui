import torch.optim
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import *

# 准备数据集
from torch.utils.data import DataLoader, dataloader

train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 长度 length
train_data_size = len(train_data)
test_data_size = len(test_data)

# 如果train_data_size=10,则训练数据集的长度为10
print("训练数据集的长度为:{}".format(train_data_size))
print("册数数据集的长度为:{}".format(test_data_size))

# 利用Dataloader来加载数据集batch_size的大小和数据集大小有关且和显存大小有关
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Mynet()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
#learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 以上完成，这里开始设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step =0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./logs/logs_train")




# 其他的可视化诸如混淆矩阵什么的你慢慢积累方法，或者问chatgpt和copilot都可以，花点钱提升效率没什么。
# 其他更加规范的训练log写法可以看看别人的项目怎么搞的，不着急
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    train_bar=tqdm(train_dataloader)
    # 训练步骤开始
    tudui.train()
    for imgs, targets in train_bar:
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_bar.desc = ("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
        #train_bar.colour = ('#00ff00')
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            #print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    tudui.eval() # 文档中提到这里和tudui.train一样只对 dropout层和BN层有用，当然你也可以设置的的自定义层
    # 测试步骤开始
    # 测试单元（可以问chatgpt怎么写一个测试单元，待考证。。。）
    # 保证测试的时候没有调优
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    # 真是没keras和tensorflow方便，那个有history的导出和自动可视化，这个可视化的代码我还有，之后可以整理到师姐的项目中
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的acc:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_acc", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    #保存模型
    if i % 5 == 0:
        torch.save(tudui, "tudui_{}.pth".format(i))
        print("tudui_{}.pth模型已保存".format(i))

writer.close()
