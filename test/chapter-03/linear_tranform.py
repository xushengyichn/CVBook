import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载图像 - 使用跨平台路径
current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, "..", "..", "imgs", "book_cover.jpg")
img = cv2.imread(img_path)

if img is None:
    print(f"无法加载图像: {img_path}")
    exit()

# 转换为RGB格式（OpenCV默认使用BGR）
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 获取图像尺寸
height, width = img.shape[:2]

# 定义线性变换矩阵（仿射变换）

# 示例1: 旋转变换 (30度)
angle = 30
center = (width / 2, height / 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

# 示例2: 欧式变换
tx=width * 0.1  # x方向平移
ty=height * 0.1 # y方向平移
# euclidean_matrix = np.array([
#     [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), tx],
#     [np.sin(np.radians(angle)), np.cos(np.radians(angle)), ty]
# ], dtype=np.float32)

euclidean_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
euclidean_matrix[:, 2] += [tx, ty]

# 示例3: 相似变换
scale_factor = 0.2
similarity_matrix = np.array([
    [scale_factor*np.cos(np.radians(angle)), -scale_factor*np.sin(np.radians(angle)), tx],
    [scale_factor*np.sin(np.radians(angle)), scale_factor*np.cos(np.radians(angle)), ty]
], dtype=np.float32)

similarity_matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
similarity_matrix[:, 2] += [tx, ty]

# 示例4: 仿射变换
affine_matrix = np.array([
    [1, 0.2, tx],
    [0.2, 1, ty]
], dtype=np.float32)

w, h = width, height

src = np.float32([
    [0, 0],
    [w, 0],
    [0, h]
])

dst = np.float32([
    [0 + tx,   0 + ty],
    [w + tx,   0 + ty + 0.2*w],   # 在 y 方向加一点“随 x 增长”的偏移 → 类似剪切
    [0 + tx + 0.2*h, h + ty]      # 在 x 方向加一点“随 y 增长”的偏移
])

affine_matrix = cv2.getAffineTransform(src, dst)

# 示例5：射影变换
projection_matrix = np.array([
    [1, 0.2, tx],
    [0.2, 1, ty],
    [0.001, 0.001, 1]
], dtype=np.float32)


w, h = width, height

src = np.float32([
    [0, 0],
    [w, 0],
    [w, h],
    [0, h]
])

dst = np.float32([
    [0 + tx,      0 + ty],
    [w + tx,      0 + ty + 0.1*h],
    [w + tx - 0.1*w, h + ty],
    [0 + tx + 0.1*w, h + ty - 0.1*h]
])

projection_matrix = cv2.getPerspectiveTransform(src, dst)

# 应用变换
img_rotated = cv2.warpAffine(img_rgb, rotation_matrix, (width, height))
img_euclidean = cv2.warpAffine(img_rgb, euclidean_matrix, (width, height))
img_similar = cv2.warpAffine(img_rgb, similarity_matrix, (width, height))
img_affine = cv2.warpAffine(img_rgb, affine_matrix, (width, height))
img_projected = cv2.warpPerspective(img_rgb, projection_matrix, (width, height))

# 显示结果
fig, axes = plt.subplots(2, 3, figsize=(6, 8))
fig.suptitle('图像线性变换示例', fontsize=16)

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(img_rotated)
axes[0, 1].set_title('旋转变换 (30度)')
axes[0, 1].axis('off')

axes[1, 0].imshow(img_euclidean)
axes[1, 0].set_title('欧式变换')
axes[1, 0].axis('off')

axes[1, 1].imshow(img_similar)
axes[1, 1].set_title('相似变换')
axes[1, 1].axis('off')

axes[0, 2].imshow(img_affine)
axes[0, 2].set_title('仿射变换')
axes[0, 2].axis('off')

axes[1, 2].imshow(img_projected)
axes[1, 2].set_title('射影变换')
axes[1, 2].axis('off')
plt.tight_layout()
plt.show()

print("线性变换完成！")
