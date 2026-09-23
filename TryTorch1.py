#2.4节微积分
#2.4.1导数和微分
#如何正确的定义函数
%matplotlib inline
from matplotlib_inline import backend_inline
from mxnet import np, npx
from d2l import mxnet as d2l

npx.set_np() #npx.set_np(shape=True, array=True)   # 返回打开之前的旧状态
# 约等于npx.set_np_shape(True) + npx.set_np_array(True)


def f(x):
    return 3 * x ** 2 - 4 * x  #`npx` = __numpy_extension__（NumPy 扩展）
#对于pytorch格式则是
%matplotlib inline
import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l


def f(x):
    return 3 * x ** 2 - 4 * x
  #然后再对求积分的运算进行定义
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
#f'...'	  f-string（Python 3.6+ 的格式化字符串字面量）。前缀 f 表示"这个字符串里的 {} 要当表达式求值再替换"
#h=、, numerical limit=	 普通字面量文本，原样打印，只是给读者看的标签
#{h:.5f}	第一个替换字段（replacement field）
#{numerical_lim(f, 1, h):.5f}	第二个替换字段，里面是一个函数调用表达式（f-string 的 {} 里允许写任意合法的 Python 表达式，不只是变量名）
#:	分隔符：冒号左边是"算什么都什么"，右边是"怎么显示"
#.5f	格式规范（format spec）：.5 = 保留 5 位小数，f = fixed-point（定点浮点，也就是普通的 2.30000 这种十进制写法）

#注释#@save是一个特殊的标记，会将对应的函数、类或语句保存在d2l包中。 因此，以后无须重新定义就可以直接调用它们（例如，d2l.use_svg_display()）。
def use_svg_display():  #@save
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):  #@save
    """设置matplotlib的图表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
   # 下面的set_axes函数用于设置由matplotlib生成图表的轴的属性。

#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
   # 通过这三个用于图形配置的函数，定义一个plot函数来简洁地绘制多条曲线， 因为我们需要在整个书中可视化许多曲线。

#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#这里绘制的是函数及其切线
    x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])

#现在进行2.5节，关于自动微分


#举例，我们想对y=2x_T *x
import torch

x = torch.arange(4.0)  #tensor([0., 1., 2., 3.])
x
#为x分配了一部分空间


x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值是None
#这个操作是因为需要分配一个地方来储存梯度


y = 2 * torch.dot(x, x)
y

#x是一个长度为4的向量，计算x和x的点积，得到了我们赋值给y的标量输出。
#接下来，通过调用反向传播函数来自动计算y关于x每个分量的梯度，并打印这些梯度。
#这里的计算结果应该是x.grad == 4 * x，我们进行验证
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad


#2.5.3分离计算举例

"""有时，我们希望将某些计算移动到记录的计算图之外。 
例如，假设y是作为x的函数计算的，而z则是作为y和x的函数计算的。 
想象一下，我们想计算z关于x的梯度，但由于某种原因，希望将y视为一个常数， 并且只考虑到x在y被计算后发挥的作用。

这里可以分离y来返回一个新变量u，该变量与y具有相同的值， 但丢弃计算图中如何计算y的任何信息。 
换句话说，梯度不会向后流经u到x。 因此，下面的反向传播函数计算z=u*x关于x的偏导数，同时将u作为常数处理， 而不是z=x*x*x关于x的偏导数。"""

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u


#由于记录了y的计算结果，我们可以随后在y上调用反向传播， 得到y=x*x关于的x的导数，即2*x。

x.grad.zero_()
y.sum().backward()
x.grad == 2 * x











"""这两段是 D2L 里的一组对照实验：同样一个 x，同样数值的 y，一段 detach 一段不 detach，看梯度差在哪。我全部实跑了（torch 2.14 CPU），下面引用的都是真实输出。

━━━━━━━━━━━━━━━━━━━━
零、先记住一个设定

    x = torch.arange(4.0, requires_grad=True)    # [0, 1, 2, 3]

后面所有计算都基于它。两段之间用 x.grad.zero_() 把梯度清零，所以它们互不干扰 —— 第二段的答案不会被第一段污染。

━━━━━━━━━━━━━━━━━━━━
一、第一段：detach 之后，梯度变成了 u

    x.grad.zero_()        # 先把梯度清零（下面第三节会讲为什么必须清）
    y = x * x             # y = [0, 1, 4, 9]，requires_grad=True，grad_fn = MulBackward0
    u = y.detach()        # u = [0, 1, 4, 9]，requires_grad=False，grad_fn = None  ← 关键一行
    z = u * x             # z = [0, 1, 8, 27]，数值上就是 x³
    z.sum().backward()    # x.grad = [0, 1, 4, 9]  ← 等于 u
    x.grad == u           # tensor([True, True, True, True])

最关键的一行是 u = y.detach()。它的效果不是「复制一份数据」，而是「把这块数据从计算图上剪下来」：

    y：值是 [0,1,4,9]，身份是「x·x 的结果」——它记得自己从哪来，grad_fn 指向 MulBackward0
    u：值是 [0,1,4,9]，身份是「一个常数」——grad_fn 为 None，autograd 从此不管它的来历

所以当程序执行 z = u * x 时，autograd 看到的是「常数 × x」，不是「x² × x」。乘法的局部导数是「把另一个因子拿过来」：

    ∂z/∂x = u        （因为 ∂(u·x)/∂x = u，把 u 当常数）

于是 x.grad = u = [0, 1, 4, 9]。

这里有一个必须点破的细节：z 的数值确实等于 x³，而 x³ 的数学导数是 3x² = [0, 3, 12, 27]。所以 detach 之后算出来的梯度，和「函数值的数学导数」是不一致的。

    实测对比：
      detach 后    x.grad = [0.0, 1.0, 4.0, 9.0]     ← = u = x²
      数学上应该   3x²     = [0.0, 3.0, 12.0, 27.0]

这不是 bug，这正是 detach 的语义：它让 autograd 假装「u 那段历史和 x 无关」。你要的就是这个效果时，它是对的；你不小心漏加或乱加时，梯度就悄悄错了。

━━━━━━━━━━━━━━━━━━━━
二、第二段：不 detach，梯度就是正常的 2x

    x.grad.zero_()        # 把上一段的 [0,1,4,9] 清零 → [0,0,0,0]
    y.sum().backward()    # x.grad = [0, 2, 4, 6]
    x.grad == 2 * x       # tensor([True, True, True, True])

这一次 y 还完好地连着图，y = x·x 的局部导数是 ∂y/∂x = 2x（乘法把两个因子互相拿给对方），所以：

    x.grad = 2x = [0, 2, 4, 6]

两段并排看，差别只有一个：

    z = u * x           → x.grad = [0, 1, 4, 9]    = x²      （u 被当作常数）
    y = x * x → y.sum() → x.grad = [0, 2, 4, 6]    = 2x      （y 的来历被记住）
    ─────────────────────────────────────────────────────────────
    同样的数、同样的 x，差别只在于「这一支的来历有没有被记在图上」

顺带说一句：y.sum().backward() 里的 .sum() 正是上一轮讲的东西 —— y 是向量，必须先压成标量（等价于传一个全 1 的梯度）。

━━━━━━━━━━━━━━━━━━━━
三、x.grad.zero_() 这一行为什么两段都有

    · 梯度是累加的（上一轮讲过）。第一段已经把 [0,1,4,9] 写进 x.grad 了；
      不清零的话，第二段会在这个基础上再加一份 [0,2,4,6]，你就读不出「第二段的梯度到底是多少」。
    · 真实训练里对应的是每个 step 之前的 optimizer.zero_grad()。这里因为没建优化器，所以手动清。

一个实用的坑（实测报错原文）：

    x.grad 为 None 时调用 x.grad.zero_()
      → AttributeError: 'NoneType' object has no attribute 'zero_'

    第一次 backward 之前 x.grad 就是 None，所以书中几段代码都先 backward 过、或用 x.grad = None 来清。
    两种清法的区别：x.grad.zero_() 把已有张量填 0（保留内存），x.grad = None 直接扔掉（PyTorch 官方更推荐后者，
    它对 set_to_none=True 的说法是省内存、也少一次写操作）。

━━━━━━━━━━━━━━━━━━━━
四、detach 到底是「复制」还是「共用」（实测）

    u 的值和 y 完全一样               : True
    u 和 y 共用同一块内存（data_ptr 相同）: True

也就是说 detach 返回的是同一个存储的视图，你改 u 的数据，y 的数据也跟着变（u 不参与求导，所以这种原地改不会报 in-place 错误，而是更隐蔽地污染 y）。要一份真正独立的副本，用

    u = y.detach().clone()

顺带把 detach() 和 with torch.no_grad() 分清：

    tensor.detach()     针对「一个张量」：把这个节点从图上剪掉，其余部分照旧建图
    with torch.no_grad() 针对「一段代码」：这段里产生的所有张量都不带图（推理、更新参数时用）

━━━━━━━━━━━━━━━━━━━━
五、什么时候你会故意 detach（这才是它存在的理由）

  1. 标签 / 目标值：label 就该是常数，不该参与求导。实测那个反面例子：
       忘记 detach 标签: w.grad = 30.0    target.grad = -10.0   ← 梯度跑到标签上去了
       detach 标签后   : w.grad = 30.0    target.grad = None    ← 干净
     （w.grad 数值一样，因为标签本身不参与 w 的导数；但标签被写上了梯度，一旦它又出现在别处就会出错。）

  2. 只想记录 loss 又想省显存：loss.detach().item() 打印/记录，不留计算图。

  3. 目标网络 / EMA 教师 / stop-gradient：DQN 的 target network、Mean Teacher、BYOL 这类结构，
     本质就是「希望某一路的梯度在这里停住」，写出来就是 .detach()。

  4. 张量转 numpy / 存盘：先 detach() 再 .numpy()（共享内存时最好 .cpu().detach().numpy()）。

  5. 排错时的第一反应：如果某个参数「不更新」或「更新得很怪」，先顺着 loss 往回找有没有 detach 或 no_grad 挡住了路 —— 这一节的两段代码就是那个现象的最小复现：数值看起来完全正常（z = x³），梯度却已经不是你想象的那个了。

━━━━━━━━━━━━━━━━━━━━
六、一句话总结

  detach() = 把这一支的「来历」抹掉，让它在这张图上变成一个常数。
  · 值不变，梯度变了：z 数值上是 x³，但 x.grad = x²，不是 3x²；
  · 它不是复制，是同一块内存的另一个「身份」；要副本得 .clone()；
  · 你在做「stop-gradient」时它是工具，你在「参数不更新」时它是嫌疑人。
"""
