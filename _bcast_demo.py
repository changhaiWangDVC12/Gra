import torch

out = []


def show(title, fn):
    out.append("=" * 70)
    out.append(title)
    try:
        fn()
    except Exception as e:
        out.append("!! %s: %s" % (type(e).__name__, e))


def ex1():
    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(2).reshape((1, 2))
    out.append("a = %s  shape=%s" % (a.tolist(), tuple(a.shape)))
    out.append("b = %s  shape=%s" % (b.tolist(), tuple(b.shape)))
    out.append("a + b =\n%s\nshape=%s" % (a + b, tuple((a + b).shape)))
    out.append("expand: a -> %s, b -> %s" % (tuple(a.expand(3, 2).shape), tuple(b.expand(3, 2).shape)))


def ex2():
    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(2)
    out.append("a shape=%s, b shape=%s" % (tuple(a.shape), tuple(b.shape)))
    out.append("a + b shape=%s\n%s" % (tuple((a + b).shape), a + b))


def ex3():
    a = torch.arange(3).reshape((1, 3))
    b = torch.arange(3)
    out.append("a shape=%s, b shape=%s -> a+b shape=%s" % (tuple(a.shape), tuple(b.shape), tuple((a + b).shape)))


def ex4():
    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(4).reshape((1, 4))
    out.append("(3,1)+(1,4) -> shape=%s" % (tuple((a + b).shape),))


def ex5():
    a = torch.arange(6).reshape((3, 2, 1))
    b = torch.arange(4).reshape((1, 4))
    out.append("(3,2,1)+(1,4): 先右对齐补 1 -> (1,3,2,1) 与 (1,1,1,4) -> shape=%s" % (tuple((a + b).shape),))
    out.append("%s" % (a + b,))


def ex6():  # 不可广播
    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(8).reshape((2, 4))
    try:
        a + b
        out.append("居然没报错? shape=%s" % tuple((a + b).shape))
    except Exception as e:
        raise


def ex7():  # 标量
    x = torch.arange(6, dtype=torch.float32).reshape((2, 3))
    y = 2.0
    out.append("x shape=%s + 标量 -> shape=%s\n%s" % (tuple(x.shape), tuple((x + y).shape), x + y))


def ex8():  # 广播的反例：长度为3的轴 vs 长度为1
    a = torch.arange(3).reshape((1, 3))
    b = torch.arange(2).reshape((1, 2))
    try:
        a + b
        out.append("没报错 shape=%s" % tuple((a + b).shape))
    except Exception as e:
        raise


def ex9():  # expand 只读视图 vs repeat 真复制
    a = torch.arange(3).reshape((3, 1))
    e = a.expand(3, 4)
    r = a.repeat(1, 4)
    out.append("expand shape=%s strides=%s is_view(共享内存)=%s" % (tuple(e.shape), e.stride(), e.data_ptr() == a.data_ptr()))
    out.append("repeat shape=%s strides=%s is_view(共享内存)=%s" % (tuple(r.shape), r.stride(), r.data_ptr() == a.data_ptr()))


def ex10():  # 逐元素运算里广播实际发生的例子：归一化
    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    mean = X.mean(dim=0)          # shape (4,)
    std = X.std(dim=0)            # shape (4,)
    out.append("X shape=%s mean shape=%s" % (tuple(X.shape), tuple(mean.shape)))
    out.append("(X - mean) / std =\n%s" % ((X - mean) / std,))


for name, t, fn in [
    ("例0 D2L 经典 (3,1)+(1,2)", "ex1", ex1),
    ("例1 (3,1)+(2,)", "ex2", ex2),
    ("例2 (1,3)+(3,)", "ex3", ex3),
    ("例3 (3,1)+(1,4)", "ex4", ex4),
    ("例4 三维 (3,2,1)+(1,4)", "ex5", ex5),
    ("例5 不可广播 (3,1)+(2,4)", "ex6", ex6),
    ("例6 标量广播", "ex7", ex7),
    ("例7 不可广播 (1,3)+(1,2)", "ex8", ex8),
    ("例8 expand vs repeat", "ex9", ex9),
    ("例9 实际应用：列标准化", "ex10", ex10),
]:
    show(name, fn)

with open("d:/gd/_bcast.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
out.append("torch=%s" % torch.__version__)
