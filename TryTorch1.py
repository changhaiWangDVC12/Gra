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
