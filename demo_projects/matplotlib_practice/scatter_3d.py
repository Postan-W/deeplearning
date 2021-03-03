import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create the figure空白图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 点数
n = 250

# 这个函数返回的是介于最大最小值之间的数的列表
f = lambda minval, maxval, n: minval + (maxval - minval) * np.random.rand(n)

# Generate the values
x_vals = f(15, 41, n)
y_vals = f(-10, 70, n)
z_vals = f(-52, -37, n)

# Plot the values
ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o',s=80)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()
