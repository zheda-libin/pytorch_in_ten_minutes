### 1. 关于交叉熵损失

* 交叉熵损失函数的API是，nn.CrossEntropyLoss()。

* 它的输入一个是logit（也就是刚刚经过FC层，还没经过ReLU以及Softmax层），另外一个是target，注意是以scaler形式为输出的（非one-hot形式）。

* 交叉熵损失函数，等价于nn.LogSoftmax() + nn.NLLLoss()。

解释：LogSoftmax，相当于，先做softmax，再取log，然后再根据target，做一步gather。

链接：[nn.CrossEntropyLoss()](https://pytorch.org/docs/1.4.0/nn.html?highlight=crossentropy#torch.nn.CrossEntropyLoss)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

logit = torch.randn(10, 4)
target = torch.randint(4, (10,))

# ===========================================
# 1. CrossEntropy Loss
loss_fn = nn.CrossEntropyLoss()
loss1 = loss_fn(logit, target)
# -------------------------------------------
# 2. NLL Loss
log_softmax_logit = F.log_softmax(logit)
loss_fn2 = nn.NLLLoss()
loss2 = loss_fn2(log_softmax_logit, target)
# -------------------------------------------
# 3. Manually
logit1 = F.softmax(logit)       # softmax version
logit2 = torch.log(logit1)      # log of softmax version
logit3 = (-1) * logit2          # negative log of softmax version.
selected = torch.gather(logit3, dim=1, 
                        index=target.unsqueeze(1))
loss3 = torch.mean(selected)
```

### 2. 关于随机种子

为保证实验的复现性，需要设置如下的随机种子：

```python
import random
import numpy as np
import torch

seed = 688
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# if torch.cuda.available = True
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```


### 3. 各种Normalization，各种N（BN, LN, IN, AdaIN）
* [nn.BatchNorm2d()](https://pytorch.org/docs/1.4.0/nn.html?highlight=batchnorm#torch.nn.BatchNorm2d)
* [nn.InstanceNorm2d()](https://pytorch.org/docs/1.4.0/nn.html#instancenorm2d)
* [nn.LayerNorm()](https://pytorch.org/docs/1.4.0/nn.html#layernorm)

1. 假设一张feature map的形状为N x C x H x W。

BN计算每一个channel的均值和方差，也就是均值和方差的形状都是C，然后用他们进行归一化。

IN计算每个样本，每个channel的均值和方差，也就是均值和方差的形状都是N x C，然后用他们进行归一化。

LN计算每个样本的均值和方差，也就是他们的shape应该是N。当然再Pytorch的API中，也可以特殊地制定对哪一个dimension进行normalize。

2. affine选项为True的时候，那么会有可学习的参数。

3. 当track_running_stat选项为True，那么在每一次正向传播过程中，都会维护一个"历史"的均值和方差。

关于tracking_running_stat和.train()和.eval()的问题，首先，tracking_running_stat设置为True，在两种情况下都对，其次，当tracking_runing_stat=False的时候，在训练阶段没有差别，在测试阶段，由于只能用当前batch或者当前一张图片的统计值，因此结果会比较差。


### 4. detach和data有什么区别？

两个都可以将一个变量从当前的计算图中剥离，区别在于，detach会有报错提醒，而data不会。

detach声明，一个variable不在当前计算图上，实际上是新建了一个变量，这个变量与原变量共享同一块数据，但是requires_grad为false，

detach通常用在，你在计算一些indices，或者计算一些label，显然，这一过程不应该是反向传播的。

对一个detach版变量，应用in-place操作，无疑会更改共享内存，也因此，这一操作会报错。

如果你就是想这么做，比如在做weight的初始化的时候，那么请使用.data。


### 5. view和reshape有什么区别

view改变的是元数据的stride，即怎么看一个tensor，如果之前使用了transpose或者permute，一定要先调用一个contiguous()函数。

而reshape则可以看成是contiguous() + view()


### 6. scheduler里面get_lr有bug

确实是有bug，这个里面比较复杂，很多PR都谈到了这一点，比如#26423, #31125, #31871。

最好使用get_last_lr进行替代。

另外scheduler.step()的步骤也很讲究。

一般是整个epoch的train结束以后，再调用一次scheduler.step()。


### 7. expand和repeat的区别

expand不会分配新的内存，只是创建一个新的视图，而repeat则会拷贝这个数据。

如果这个里面说expand一个数据，目标的扩增为-1的话，那么就是不扩增。

### 8. module和children有什么不同

children只返回第一层子模块，后面再有后代，就不会返回了，而module则是深度有限搜索，递归地返回所有内容。

例子：

[[1, 2], 3, 4]

children遍历：[1, 2], 3, 4
module遍历：[[1, 2], 3, 4], [1, 2], 1, 2, 3, 4



