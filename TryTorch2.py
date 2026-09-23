#2.6概率
#2.6.1基本概率论


#这是一些必要的软件包
%matplotlib inline
import torch
from torch.distributions import multinomial
from d2l import torch as d2l

#为了抽取样本，我们需要传入一个概率向量，输出是另一个相同长度的向量
fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()  #模拟掷 1 次骰子，返回每个面出现次数的向量

"""
fair_probs = torch.ones([6]) / 6
拆解步骤一：
torch.ones([6])
生成一个长度为 6 的全 1 张量：
tensor([1., 1., 1., 1., 1., 1.])
拆解步骤二：
再除以 6
fair_probs = tensor([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
它表示一个 公平六面骰子：每个面出现的概率都是 1/6。"""


multinomial.Multinomial(10, fair_probs).sample()  #使用深度学习框架的函数同时抽取10个样本，得到我们想要的任意形状的独立样本数组。

# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000  # 相对频率作为估计值
#搞了1000个样本，我们知道这个数应该很接近六分之一，即约0.167

#下面是一个进行了500组实验的例子，每组抽取10个样本
counts = multinomial.Multinomial(10, fair_probs).sample((500,))  #.sample((500,))：表示做 500 组这样的实验。
#所以我们产生的结果的结构是：(500, 6)
#其中的六列分别表示10次抽样后六面中该面出现的次数，例如[2., 1., 3., 1., 2., 1.]

cum_counts = counts.cumsum(dim=0)  #只剩行，对每一列进行大合并
#表现形式是cum_counts[i] 表示：
#前 i+1 组实验加起来，每个面一共出现了多少次。
#但是它本质上和原函数的格式相同，不过为累加求和形式
"""
counts[0]      = [2, 1, 3, 1, 2, 1]
counts[1]      = [1, 2, 1, 3, 0, 3]

cum_counts[0] = [2, 1, 3, 1, 2, 1]
cum_counts[1] = [3, 3, 4, 4, 2, 4]
"""
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

#estimates[i, j]
#表示：前 i+1 组（从0到i，并且包括i，这一点看上面的counts[1]就已经有所体现)实验后，第 j 个面出现的估计概率。

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();