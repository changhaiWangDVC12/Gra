import torch

out = []
a = torch.arange(3).reshape((3, 1))
e = a.expand(3, 4)
out.append("e.stride()=%s" % (e.stride(),))
try:
    e[0, 0] = 99
    out.append("expand 结果可原地写入? 没报错")
except Exception as ex:
    out.append("expand 结果原地写入 -> %s: %s" % (type(ex).__name__, ex))

out.append("torch.broadcast_shapes((3,1),(1,4)) = %s" % (torch.broadcast_shapes((3, 1), (1, 4)),))
try:
    out.append("torch.broadcast_shapes((3,1),(2,4)) = %s" % (torch.broadcast_shapes((3, 1), (2, 4)),))
except Exception as ex:
    out.append("torch.broadcast_shapes((3,1),(2,4)) -> %s: %s" % (type(ex).__name__, ex))

x = torch.ones(3, 4)
y = torch.arange(3, dtype=torch.float32).reshape((3, 1))
x += y  # in-place 广播
out.append("in-place 广播 x += (3,1):\n%s" % x)

with open("d:/gd/_bcast2.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
