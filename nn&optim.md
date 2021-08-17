`torch.nn`是专门为神经网络设计的**模块化接口**。`nn`构建于 `Autograd`之上，可用来**定义和运行神经网络**。 这里我们主要介绍几个一些常用的类

约定：`torch.nn` 我们为了方便使用，会为他设置别名为`nn`，本章除`nn`以外还有其他的命名约定

除了`nn`别名以外，我们还引用了`nn.functional`，这个包中包含了神经网络中使用的一些**常用函数**，这些函数的特点是，不具有可学习的参数(如ReLU，pool，DropOut等)，这些函数可以放在构造函数中，也可以不放，但是这里建议不放。

一般情况下我们会将`nn.functional`设置为大写的`F`，这样缩写方便调用
即
```python
# 首先要引入相关的包
import torch
# 引入torch.nn并指定别名
import torch.nn as nn
import torch.nn.functional as F
```

# 定义一个网络
PyTorch中已经为我们准备好了现成的网络模型，只要继承`nn.Module`，并实现它的`forward`方法，PyTorch会根据`autograd`，自动实现`backward`函数，在`forward`函数中可使用任何tensor支持的函数，还可以使用`if`、`for`循环、`print`、`log`等Python语法，写法和标准的Python写法一致。
```python
class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()
        # 卷积层 '1'表示输入图片为单通道， '6'表示输出通道数，'3'表示卷积核为3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 线性层，输入1350个特征，输出10个特征
        self.fc = nn.Linear(1350, 10)

    # 正向传播
    def forward(self, x):
        print(x.size())  # 结果：[1, 1, 32, 32]
        # 卷积 -> 激活 -> 池化
        x = self.conv1(x)  # 根据卷积的尺寸计算公式，计算结果是30，具体计算公式后面第二章第四节 卷积神经网络 有详细介绍。
        x = F.relu(x)
        print(x.size())  # 结果：[1, 6, 30, 30]
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(x)
        print(x.size())  # 结果：[1, 6, 15, 15]
        # reshape，‘-1’表示自适应
        # 这里做的就是压扁的操作 就是把后面的[1, 6, 15, 15]压扁，变为 [1, 1350]
        x = x.view(x.size()[0], -1)
        print(x.size())  # 这里就是fc1层的的输入1350
        x = self.fc(x)
        return x


net = Net()
print(net)
```
输出
```
Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (fc): Linear(in_features=1350, out_features=10, bias=True)
)
```
网络的可学习参数通过`net.parameters()`返回
```python
for parameters in net.parameters():
    print(parameters)
```
输出
```
Parameter containing:
tensor([[[[ 0.1624, -0.1307,  0.2517],
          [-0.1018, -0.0964,  0.2792],
          [ 0.0973,  0.0310, -0.2952]]],


        [[[ 0.2555, -0.0830,  0.0927],
          [ 0.2684,  0.2079, -0.1449],
          [-0.2507, -0.1730, -0.1439]]],


        [[[ 0.3002, -0.1717, -0.0733],
          [ 0.1629,  0.2670,  0.0715],
          [-0.1396,  0.2412,  0.1074]]],


        [[[ 0.1405,  0.1157, -0.1615],
          [ 0.2375,  0.0771,  0.1884],
          [ 0.1856,  0.1810, -0.0995]]],


        [[[ 0.2237,  0.2253,  0.0392],
          [ 0.3330, -0.0812, -0.0758],
          [ 0.1690, -0.2413, -0.0928]]],


        [[[ 0.0620,  0.0744,  0.2831],
          [-0.0923, -0.3167, -0.1067],
          [-0.0456, -0.2949, -0.1036]]]], requires_grad=True)
Parameter containing:
tensor([-0.2227,  0.0707, -0.1218,  0.1798,  0.2694,  0.0908],
       requires_grad=True)
Parameter containing:
tensor([[ 0.0217,  0.0086, -0.0121,  ..., -0.0218, -0.0124, -0.0170],
        [-0.0114, -0.0087, -0.0091,  ...,  0.0116, -0.0144, -0.0082],
        [-0.0180,  0.0219, -0.0144,  ..., -0.0205,  0.0119,  0.0202],
        ...,
        [-0.0116,  0.0161, -0.0109,  ...,  0.0057, -0.0191, -0.0169],
        [ 0.0218, -0.0103, -0.0250,  ...,  0.0109, -0.0059,  0.0056],
        [ 0.0050,  0.0073, -0.0194,  ..., -0.0030, -0.0071, -0.0169]],
       requires_grad=True)
Parameter containing:
tensor([-0.0171, -0.0120, -0.0149,  0.0236,  0.0071,  0.0108, -0.0198, -0.0047,
        -0.0226,  0.0131], requires_grad=True)
```

`net.named_parameters`可同时返回可学习的参数及名称。
```python
for name, parameters in net.named_parameters():
    print(name, ':', parameters.size())
```
输出
```
conv1.weight : torch.Size([6, 1, 3, 3])
conv1.bias : torch.Size([6])
fc.weight : torch.Size([10, 1350])
fc.bias : torch.Size([10])
```

forward函数的输入和输出都是Tensor

```python
input = torch.randn(1, 1, 32, 32) # 这里的对应前面forward的输入是32
out = net(input)
print(out.size())
```



输出
```
torch.Size([1, 1, 32, 32])
torch.Size([1, 6, 30, 30])
torch.Size([1, 6, 15, 15])
torch.Size([1, 1350])
torch.Size([1, 10])
```

```
print(input.size())
```
输出
```
torch.Size([1, 1, 32, 32])
```

在反向传播前，先要将所有参数的梯度清零
```
net.zero_grad() 
out.backward(torch.ones(1,10)) # 反向传播的实现是PyTorch自动实现的，我们只要调用这个函数即可
```

注意:`torch.nn`只支持mini-batches，不支持一次只输入一个样本，即一次必须是一个batch。

也就是说，就算我们输入一个样本，也会对样本进行分批，所以，所有的输入都会增加一个维度，我们对比下刚才的input，nn中定义为3维，但是我们人工创建时多增加了一个维度，变为了4维，最前面的1即为batch-size

# 损失函数
在`nn`中PyTorch还预制了常用的损失函数，下面我们用MSELoss用来计算均方误差
```python
y = torch.arange(0,10).view(1,10).float()
criterion = nn.MSELoss()
loss = criterion(out, y)
#loss是个scalar，我们可以直接用item获取到他的python类型的数值
print(loss.item())
```
输出
```
26.622814178466797
```

# 优化器
在反向传播计算完所有参数的梯度后，还需要使用优化方法来更新网络的权重和参数，例如随机梯度下降法(SGD)的更新策略如下：

weight = weight - learning_rate * gradient

在`torch.optim`中实现大多数的优化方法，例如RMSProp、Adam、SGD等，下面我们使用SGD做个简单的样例

```python
import torch.optim
#新建一个优化器，SGD只需要要调整的参数和学习率
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
# 先梯度清零(与net.zero_grad()效果一样)
optimizer.zero_grad()
loss.backward()
#更新参数
optimizer.step()
```