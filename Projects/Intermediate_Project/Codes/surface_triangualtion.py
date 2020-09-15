from mpl_toolkits import mplot3d
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# A simple sine function
def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

# Surface Triangulation
theta = 2 * np.pi * np.random.random(1000)
r = 6 * np.random.random(1000)
x = np.ravel(r * np.sin(theta))
y = np.ravel(r * np.cos(theta))
z = f(x,y)
ax = plt.axes(projection='3d')
ax.plot_trisurf(x,y,z,cmap='viridis',edgecolor='none')
# ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5) # Plot the above trisurf as a scatter
plt.show()
