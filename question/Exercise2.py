"""练习 2：间距矩阵与最小距离曲线
算一个 100 × 100 的两车距离矩阵（第 i 帧的 A 车对第 j 帧的 B 车）——目的是练广播机制
取「同帧」距离，得到 100 个数的 d(t)，画成曲线（横轴时间 s，纵轴距离 m）
顺手算一下 TTC = d / Δv（Δv 是两车速度差），把这条曲线也画在同一张子图上"""
import numpy as np
import matplotlib.pyplot as plt

fps = 10
n = 100
t = np.arange(n) / fps    #与第一题时间与帧相同
#车A：本车道y=0，速度为15m/s，初始位置x=0
vA = 15.0
yA = 0.0
xA = vA * t                     # (100,)，是距离坐标计算公式

# 车 B：在相邻车道y=3.5，12 m/s，初始在 A 前方 60 m
vB = 12.0
yB = 3.5
xB = 60.0 + vB * t              # (100,)，距离，实为xA = [xA[0], xA[1], ..., xA[99]] 与xB = [xB[0], xB[1], ..., xB[99]]

# 形状 (100, 2)：第 0 列 x 位置，第 1 列 vx 速度
A = np.column_stack([xA, np.full(n, vA)])
B = np.column_stack([xB, np.full(n, vB)])


#到这里还和第一题的内容基本相同
# 2. 距离矩阵：D[i, j] = 第 i 帧 A 与第 j 帧 B 的距离
# 广播：xA[:, None] 形状 (100, 1)，xB[None, :] 形状 (1, 100)
#也即，xa现在是一列，而xb则是一行
# 相减得到 (100, 100)
#也就是说，none是“开辟新轴”来的


"""左边形状 (100, 1)，右边形状 (1, 100)。

NumPy 的广播规则是：从右往左对齐维度，如果某个维度是 1，就自动扩展成另一个数组对应维度的大小。

左边：(100, 1) → 在第二个维度上扩展成 100 → (100, 100)

右边：(1, 100) → 在第一个维度上扩展成 100 → (100, 100)

于是它们都变成了 (100, 100) 的矩阵，然后逐元素相减。"""
dx = xA[:, None] - xB[None, :]   # (100, 100)
dy = yA - yB                     # 标量 -3.5
#现在对于整个矩阵都有dx[i, j] = xA[i] - xB[j]
D = np.sqrt(dx**2 + dy**2)       # (100, 100)（勾股定理）

print("D.shape =", D.shape)      # (100, 100)

"""这里要注意区分D和d_same的问题，
D 是一个 100×100 的矩阵，包含 所有帧组合 的距离：第 i 帧的 A 车 与 第 j 帧的 B 车 之间的距离。
d_same 是一个长度为 100 的一维数组，只包含 同一帧 的距离：第 i 帧的 A 车 与 第 i 帧的 B 车 之间的距离。"""


# 3. 同帧距离 d(t)

# 对角线元素就是第 i 帧 A 与第 i 帧 B 的距离
d_same = D[np.arange(n), np.arange(n)]   # (100,)

#实为 d_same = np.sqrt((xA - xB)**2 + (yA - yB)**2)

# 4. TTC = d / Δv   time to collision

vA_arr = A[:, 1]                 # (100,)
vB_arr = B[:, 1]                 # (100,)
dv = vA_arr - vB_arr             # 速度差，这里恒为 15 - 12 = 3 m/s

ttc = d_same / dv                # (100,)



#以下部分图示设定为agent代写
# 5. 画图：同一张图中有两个纵轴
fig, ax1 = plt.subplots(figsize=(9, 4.5))

# 左轴：距离
ax1.plot(t, d_same, 'b-', label='Distance d(t)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Distance (m)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# 右轴：TTC
ax2 = ax1.twinx()
ax2.plot(t, ttc, 'r--', label='TTC = d / Δv')
ax2.set_ylabel('TTC (s)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.title('Same-frame distance and TTC')
plt.tight_layout()
plt.show()

# 6. 保存结果
np.savez('distance_ttc.npz', D=D, d_same=d_same, ttc=ttc, t=t)