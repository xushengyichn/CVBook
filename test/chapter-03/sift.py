import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Parameters for isotropic Gaussian
mu = 0.0
sigma = 1.0
size = 5.0
step = 0.05

x = np.arange(-size, size + step, step)
y = np.arange(-size, size + step, step)
X, Y = np.meshgrid(x, y)

Z = (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(-((X - mu)**2 + (Y - mu)**2) / (2.0 * sigma**2))

fig = plt.figure(figsize=(10, 4))

# 3D surface
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax1.set_title('2D Isotropic Gaussian (Surface)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# 2D contour
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X, Y, Z, levels=30, cmap='viridis')
ax2.set_title('2D Isotropic Gaussian (Contour)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
fig.colorbar(contour, ax=ax2)

plt.tight_layout()
plt.show()



# Second derivatives (Laplacian of Gaussian)
d2Z_dx2 = ((X - mu)**2 - sigma**2) / (sigma**4) * Z
d2Z_dy2 = ((Y - mu)**2 - sigma**2) / (sigma**4) * Z
LoG = d2Z_dx2 + d2Z_dy2

fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, LoG, cmap='coolwarm', edgecolor='none')
ax1.set_title('Laplacian of Gaussian (Surface)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('LoG')

ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X, Y, LoG, levels=30, cmap='coolwarm')
ax2.set_title('Laplacian of Gaussian (Contour)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
fig.colorbar(contour, ax=ax2)

plt.tight_layout()
plt.show()

# Scale-normalized LoG (sigma^2 * LoG)
LoG_norm = (sigma**2) * LoG

fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, LoG_norm, cmap='coolwarm', edgecolor='none')
ax1.set_title('Scale-Normalized LoG (Surface)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('sigma^2 * LoG')

ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X, Y, LoG_norm, levels=30, cmap='coolwarm')
ax2.set_title('Scale-Normalized LoG (Contour)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
fig.colorbar(contour, ax=ax2)

plt.tight_layout()
plt.show()