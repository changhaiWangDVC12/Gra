"""练习 1：造两条车的轨迹
用 NumPy 造两条各 100 帧（10 Hz，即 10 秒）的轨迹，含位置和速度：

车 A：在本车道（y = 0）以 15 m/s 匀速行驶
车 B：在相邻车道（y = 3.5 m）以 12 m/s 匀速行驶，初始位置在 A 前方 60 m
输出两个形状为 (100, 2) 的数组（位置与速度），存成 .npz  """
import numpy as np

fps=10 #frequency=10Hz
n_frames=100 #共100帧,1s为10帧，总计10s
t=np.arange(n_frames)/fps#除以频率，本质是记录帧，即帧的时间为0.，0.1,0.2。。。9.9s
#在这个问题中，变量只存在一维位置和速度，可以直接进行表示
#我们选定汽车前进方向为X轴，相应的，两车左右向距离即为固定y轴
#所以我们有y_a=0,y_b=3.5

#   np.column_stack  的作用是：把多个一维数组按列拼成一个二维数组。
A=np.column_stack([
    15.0*t,   #速度*时间即为位置
    np.full(n_frames,15.0) #这是速度   np.full(n_frames, 12.0) 会生成一个长度为 100 的数组，每个元素都是 12.0
])
# 车 B：初始在 A 前方 60 m，所以 x0=60, vx=12 m/s, y=3.5
B = np.column_stack([
    60.0 + 12.0 * t,          # x 位置
    np.full(n_frames, 12.0)   # vx 速度
])


#接下来将结果保存为npz格式
np.savez("car_tracks.npz",A=A,B=B)

print("A.shape=",A.shape)
print("B.shape=",B.shape)
print("A 前 3 帧:\n", A[:3])
print("B 前 3 帧:\n", B[:3])
"""A.shape= (100, 2)
B.shape= (100, 2)
A 前 3 帧:
 [[ 0.  15. ]
 [ 1.5 15. ]
 [ 3.  15. ]]
B 前 3 帧:
 [[60.  12. ]
 [61.2 12. ]
 [62.4 12. ]]  """