"""练习 3：画一张 BEV 俯视图（练习 4 加分项：做成动画）
俯视图（鸟瞰）：横轴 x（米），纵轴 y（米），比例必须 1:1，否则图的几何是错的
画两条车道线（y = 0 与 y = 3.5），画两条轨迹，用颜色深浅表示速度（scatter(c=speed, cmap=…) + colorbar）
坐标轴要带单位 m，图要有标题和图例，存成 figs/bev_day5.png
加分：用 FuncAnimation 做成沿时间播放的动图，存成 figs/bev_day5.gif"""

"""图像分层的思路：
第 1 层（静态背景）：车道线         —— 画一次，永不变
第 2 层（动态轨迹）：A/B 车走过的路径 —— 每帧“生长”一点
第 3 层（当前状态）：A/B 车当前位置点 —— 每帧移动"""

"""绘制GIF的思路：动画的本质就是
“清空画布 → 按当前帧重新画一遍 → 输出成一张图 → 重复 100 次 → 连起来变成 GIF”

FuncAnimation 就是自动循环的工具。"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter

os.makedirs('figs', exist_ok=True)

# ==========================================
# 1. 轨迹数据（同练习 1）
# ==========================================
fps = 10
n = 100
t = np.arange(n) / fps          # 0 ~ 9.9 s

vA, yA, xA0 = 15.0, 0.0, 0.0
vB, yB, xB0 = 12.0, 3.5, 60.0

xA = xA0 + vA * t               # (100,)
xB = xB0 + vB * t               # (100,)
yA_arr = np.full(n, yA)         # (100,)
yB_arr = np.full(n, yB)         # (100,)

# 速度数组（这里恒定，但保持通用：可以换成变速度）
speedA = np.full(n, vA)
speedB = np.full(n, vB)

# ==========================================
# 2. 静态 BEV 图
# ==========================================
fig, ax = plt.subplots(figsize=(12, 4))

# --- 车道线 ---
x_lane = [-10, 160]
ax.plot(x_lane, [0, 0],   'k-',  lw=1.5, alpha=0.7, label='Lane boundary')
ax.plot(x_lane, [3.5, 3.5],'k-', lw=1.5, alpha=0.7)
ax.plot(x_lane, [1.75, 1.75], 'k--', lw=0.8, alpha=0.35, label='Lane center')

# --- 轨迹点，颜色表示速度 ---
vmin, vmax = 10, 16
scA = ax.scatter(xA, yA_arr, c=speedA, cmap='autumn',
                 s=28, vmin=vmin, vmax=vmax, edgecolors='none')
scB = ax.scatter(xB, yB_arr, c=speedB, cmap='winter',
                 s=28, vmin=vmin, vmax=vmax, edgecolors='none')

# --- 颜色条（两条共用同一个色标范围）---
cbar = fig.colorbar(scA, ax=ax, pad=0.01)
cbar.set_label('Speed (m/s)')

# --- 坐标轴与比例 ---
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_aspect('equal', adjustable='box')   # ★ 1:1 比例
ax.set_xlim(-10, 160)
ax.set_ylim(-2, 6)
ax.set_title('BEV Top-Down View — trajectories colored by speed')

# --- 图例 ---
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='k', lw=1.5, alpha=0.7, label='Lane boundary'),
    Line2D([0], [0], color='k', lw=0.8, ls='--', alpha=0.35, label='Lane center'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red',
           markersize=8, label=f'Car A (y=0, v={vA} m/s)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue',
           markersize=8, label=f'Car B (y=3.5, v={vB} m/s)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('figs/bev_day5.png', dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# 3. 动画：FuncAnimation
# ==========================================
fig2, ax2 = plt.subplots(figsize=(12, 4))

# --- 车道线 ---
ax2.plot(x_lane, [0, 0],    'k-',  lw=1.5, alpha=0.7)
ax2.plot(x_lane, [3.5, 3.5],'k-',  lw=1.5, alpha=0.7)
ax2.plot(x_lane, [1.75, 1.75], 'k--', lw=0.8, alpha=0.35)

# --- 动态元素 ---
pathA, = ax2.plot([], [], '-',  color='tab:red',  lw=1.5, alpha=0.7, label='Car A path')
pathB, = ax2.plot([], [], '-',  color='tab:blue', lw=1.5, alpha=0.7, label='Car B path')
ptA,   = ax2.plot([], [], 'o',  color='tab:red',  ms=10, label='Car A')
ptB,   = ax2.plot([], [], 'o',  color='tab:blue', ms=10, label='Car B')

time_text = ax2.text(0.02, 0.92, '', transform=ax2.transAxes, fontsize=11)

ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_aspect('equal', adjustable='box')   # ★ 1:1 比例
ax2.set_xlim(-10, 160)
ax2.set_ylim(-2, 6)
ax2.set_title('BEV Animation — 10 Hz, 10 s')
ax2.legend(loc='upper right', fontsize=9)

def init():
    pathA.set_data([], [])
    pathB.set_data([], [])
    ptA.set_data([], [])
    ptB.set_data([], [])
    time_text.set_text('')
    return pathA, pathB, ptA, ptB, time_text

def update(frame):
    # 逐帧“生长”的轨迹
    pathA.set_data(xA[:frame+1], yA_arr[:frame+1])
    pathB.set_data(xB[:frame+1], yB_arr[:frame+1])
    # 当前车辆位置
    ptA.set_data([xA[frame]], [yA_arr[frame]])
    ptB.set_data([xB[frame]], [yB_arr[frame]])
    # 时间文字
    time_text.set_text(f't = {t[frame]:.1f} s')
    return pathA, pathB, ptA, ptB, time_text

anim = FuncAnimation(
    fig2, update, frames=n, init_func=init,
    blit=True, interval=100   # 100 ms = 10 Hz，与实际采样一致
)

anim.save('figs/bev_day5.gif', writer='pillow', fps=fps)
plt.close(fig2)

print('saved: figs/bev_day5.png')
print('saved: figs/bev_day5.gif')
# 方式 1：保存 GIF
anim.save('figs/bev_day5.gif', writer='pillow', fps=fps)

# 方式 2：想在窗口里播放，需要保留一个对 anim 的引用，并 plt.show()
plt.show()

"""流程图如下：
                t (100,)
                 │
   ┌─────────────┴─────────────┐
   │                           │
  xA = 15·t                  xB = 60 + 12·t
   │                           │
   └─────────────┬─────────────┘
                 │
        静态背景（画一次）：
        - 车道线 y=0, y=3.5, y=1.75
                 │
        FuncAnimation(frames=0..99)
                 │
        每帧 frame=k 调用 update(k)：
        ┌──────────────────────────────────────┐
        │ pathA ← (xA[0..k], yA[0..k])        │  轨迹生长
        │ pathB ← (xB[0..k], yB[0..k])        │
        │ ptA   ← (xA[k], yA[k])              │  当前位置
        │ ptB   ← (xB[k], yB[k])              │
        │ text  ← "t = k/10 s"                │
        └──────────────────────────────────────┘
                 │
         合成 100 帧 → GIF"""