import numpy as np
import matplotlib.pyplot as plt

# Define the number of values
num_vals = 40

# Generate random values
x = np.random.rand(num_vals)
y = np.random.rand(num_vals)

#最大半径
max_radius = 25
#随机半径下的面积
area = np.pi * (max_radius * np.random.rand(num_vals)) ** 2  

# 随机颜色
colors = np.random.rand(num_vals)

# Plot the points
plt.scatter(x, y, s=area, c=colors, alpha=1.0)

plt.show()
