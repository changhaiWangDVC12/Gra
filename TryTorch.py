import torch
#关于 n维数组，也称为张量（tensor） 
#张量类（在MXNet中为ndarray， 在PyTorch和TensorFlow中为Tensor）与Numpy的ndarray类似
x=torch.arange(12) #创建一个一维张量，包含从0到11的整数
print(x)
torch.Size([12]) #张量的形状
x.shape #张量的形状
x.numel() #张量中元素的总数
x.reshape(3,4) #将一维张量变为二维张量，3行4列
print(x)
torch.zeros((2,3,4)) #创建一个形状为(2,3,4)的全零张量
#当然还可以随机成为其他的值
torch.tensor([[1,2,3],[4,5,6]]) #创建一个二维张量（也就是直接赋值）
print(torch.tensor([[1,2,3],[4,5,6]]))


#以下内容为2.1.2节运算符及其相关内容
x=torch.tensor([[1,2,3],[4,5,6]]) #创建一个二维张量
y=torch.tensor([[7,8,9],[10,11,12]]) #创建另一个二维张量
print(x+y) #张量的加法
torch.exp(x) #张量的指数运算（好像就是变成e的x次方）
#关于多个张量的连结问题如下
 #X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
#(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [ 2.,  1.,  4.,  3.],
         [ 1.,  2.,  3.,  4.],
         [ 4.,  3.,  2.,  1.]]),
 tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))
#x.sum() #张量中所有元素的和(可以用于求和)

#以下是2.1.3节的内容
#张量的广播机制,实际上是整理成一张大网，然后复制缺少的行和列。最后再进行加法
#我们将两个矩阵广播为一个更大的矩阵，如下所示：矩阵a将复制列， 矩阵b将复制行，然后再按元素相加。
#继续到2.1.4节，张量的索引和切片
X[0:2, :] = 12
#这里指的是将X的前两行的所有列都赋值为12，即0行和1行的所有元素赋值为12
#节省内存的方式是将运算后的结果赋值给另外一个变量，而不是后面再进行重复的运算
#2.1.6 numpy数组h和张量的转换
#张量和Numpy数组之间的转换非常容易。我们可以使用torch.tensor函数
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)


#接下来从2.2.1开始，我们开始读取数据集，并存储在.csv文件中.对于这个设计来说，实际上csv就是一个按格式存储数据的文本文件，里面的每一行都是一个数据样本，每一列都是一个特征。我们可以使用pandas库来读取csv文件，并将其转换为张量。
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)#此处..可以跳转至上一级目录，而exist_ok=True表示目录如果现在已经存在也是没有问题的
data_file = os.path.join('..', 'data', 'house_tiny.csv')#此处是完整的文件路径实际上表示的是../data/house_tiny.csv
with open(data_file, 'w') as f:  #w就是之前经常说的写模式，文件不存在就直接新建，存在的话先进行清空覆盖   with ..as f就是一个高科技的上下文管理器
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
    f.write('3,NA,90000\n')

    import pandas as pd

data = pd.read_csv(data_file)#确切地说，这是一个文件格式的整理，从原来的文本格式整理为二维的带标签的表格的形式
print(data)  #这个就是一个可以打印（按照标准格式打印）的方式
#整理完成的格式类似下图这样
#NumRooms Alley   Price
#0       NaN  Pave  127500
#1       NaN   NaN  178100
#2       NaN   NaN  140000

#NumRooms    float64
#Alley           str
#Price         int64
#dtype: object
#实际上是在做规格化的统一，也就是让数据的可读性变得更好，包括对缺失符号在内的所有的标记符号进行简单的优化，例如统一转换为NaN

#接下来我们进入标题2.2.2节，学习如何处理缺失值
#解决方式主要分为两类（也就是生：插值法，死：删除法）
#插值法就是通过其他的数值来进行推测，填充缺失值，删除法就是直接删除缺失值所在的行或者列
inputs = pd.get_dummies(inputs, dummy_na=True)#这一行做的内容，实际上就是将选中范围内的数据进行简单的二元化，拆解为含有若干个0/1的指示列
#`dummy_na=True` 的作用：把"缺失"也当成一个合法类别实际上我的感觉就是把na(实际上是nan)也进行合法化
print(inputs)#可以看出，进行调整之后，数据的数量确实发生了一些变化

#接下来到达章节2.2.3节，我们如何将处理过的数据转化为张量的形式（实际上，我们仍然默认采用了pytorch的格式
#是torch.tensor而不是np.array
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
X, y
#以上进行了数据格式的经典转换，从最初的文本格式，最后总转换为了张量


#现在我们进入章节2.3 补充一下线性代数的相关问题
#在章节2.3.1中，一个数学公式内容就可以划分为标量和变量（定死的数和变量值的区别）
#y=x-2，对于这个公式来说，x和y是变量，2是标量(而且好像不是-2)
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
#现在我们已经可以进行相应的计算
#实际上的结果大致是这个样子的(tensor(5.), tensor(6.), tensor(1.5000), tensor(9.))

#接下来我们进入2.3.2关于向量
#我们说向量是很多标量的组合结果（但是从向量的角度来进行称呼，我们可以将其命名为向量的元素或者向量的分量
x = torch.arange(4)
x    #（这里实际上就是创造了一个向量）
#注意这时候下标就是从1开始的，不可能是0开始的，所以添加索引的时候要格外注意（这里存疑）

#现在我们来到2.3.2.1，涉及的问题是向量的长度，维度和形状
len(x), x.shape, x.numel() #len()函数是python的内置函数，返回对象（字符、列表、元组等）长度或元素个数
#之前我们已经学习了numel()函数，它返回张量中元素的总数
#相对的，这一期的新内容就是len()函数，它返回对象(一般在这里指向量的长度或者元素个数)
#接下来这段很重要：当用张量表示一个向量（只有一个轴）时，我们也可以通过.shape属性访问向量的长度。 
#形状（shape）是一个元素组，列出了张量沿每个轴的长度（维数）。 对于只有一个轴的张量，形状只有一个元素。
#（也就是说在每一个轴的延伸程度（类似于长宽高这样的数据）


#现在我们到达2.3.3,学习矩阵的相关知识
A = torch.arange(20).reshape(5, 4)
A #像这样就是一个经典的矩阵，5行4列
A.T#这是矩阵的转置（记得是aij=bji），也就是行列互换


X = torch.arange(24).reshape(2, 3, 4)
X#像这种情况就属于3行4列，然后在上面多加一个维度，好像就是这样

tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])

#2.3.5张量算法的基本性质
#两个矩阵的按元素乘法称为Hadamard积（Hadamard product）,具体内容不在进行演示

#2.3.6降维
#实际上就是调用求和函数进行求和（来达到降低维度的目的，而且使它成为一个标量）
x=torch.arange(4,dtype=torch.float32)
x,x.sum()# 默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量。 我们还可以指定张量沿哪一个轴来通过求和降低维度

#这里是有补充文件的，可以进行参考。
A.sum(axis=[0, 1])  #这里同时要求行和列进行求和，也就是要求最终求和
A.mean(),A.sum()#一个与求和相关的量是平均值（mean或average）。 我们通过将总和除以元素总数来计算平均值。

#下面到达2.3.6.1非降维求和
sum_A = A.sum(axis=1, keepdims=True)
sum_A  
#有补充知识文件可以进行参考
A.cumsum(axis=0)#如果我们想沿某个轴计算A元素的累积总和， 比如axis=0（按行计算），可以调用cumsum函数。 此函数不会沿任何轴降低输入张量的维度。

#接下来是2.3.7点积          
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)

#点积的计算结果是一个数，不是向量。别把它和「逐元素乘」x * y 混起来—— 后者出来还是长度为 d 的向量。
import numpy as np
A = np.arange(20, dtype=float).reshape(5, 4)
x = np.arange(4, dtype=float)


#2.3.8矩阵向量积 Ax 是一个长度为 m 的列向量， 其第 i 个元素是点积 ai⊤x
#(m, n)  ×  (n,)  →  (m,)
#中间那个 n 必须对上并被消掉：A 的列数 = x 的长度。剩下的 m 就是结果长度。
print(A.shape, x.shape)        # (5, 4) (4,)
print(np.dot(A, x))            # 等价写法之一
print(A @ x)                   # 等价写法之二
print((A * x).sum(axis=1))     # 等价写法之三
print(np.array([np.dot(A[i], x) for i in range(5)]))
#尤其是print(np.dot(A.x))这个式子可能更有助于理解
#np.dot(x, y) 要求 x 和 y 长度完全相等； 长度不等会直接报错，点积不会「自动对齐」。
#2.3.9矩阵矩阵乘法同理，不再做介绍

#现在是2.3.10具体对范数的介绍在毕设专用的文件夹内，这里只考虑代码
u = torch.tensor([3.0, -4.0])
torch.norm(u)
#这就是计算L_2范式的标准方式
#与L_2范数相比，L_1范数受异常值的影响较小。 为了计算它，我们将绝对值函数和按元素求和组合起来。
torch.abs(u).sum() #记住了啊，torch.abs(u).sum()