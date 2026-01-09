"""
详细的SIFT算法实现 - 展示每个步骤的具体操作过程
参照OpenSIFT (Rob Hess) 的C++实现
中文注释由张林老师编写，2022年8月
Python实现：2026年1月
"""

import argparse
import os
import numpy as np
import cv2
from typing import List, Tuple


# ==================== SIFT 参数定义 ====================
SIFT_INTVLS = 3              # 每组的采样间隔数（层数）
SIFT_SIGMA = 1.6             # 初始高斯平滑参数
SIFT_CONTR_THR = 0.04        # 关键点对比度阈值 |D(x)|
SIFT_CURV_THR = 10           # 关键点主曲率比阈值（用于去除边缘响应）
SIFT_DESCR_WIDTH = 4         # 描述子直方图数组宽度 (4x4)
SIFT_DESCR_HIST_BINS = 8     # 描述子每个直方图的bin数量

SIFT_INIT_SIGMA = 0.5        # 假定输入图像的高斯模糊
SIFT_IMG_BORDER = 5          # 忽略边界内的关键点的宽度
SIFT_MAX_INTERP_STEPS = 5    # 关键点插值的最大步数

SIFT_ORI_HIST_BINS = 36      # 方向分配直方图的bin数量
SIFT_ORI_SIG_FCTR = 1.5      # 方向分配的高斯sigma因子
SIFT_ORI_RADIUS = 3.0 * SIFT_ORI_SIG_FCTR  # 方向分配区域半径
SIFT_ORI_SMOOTH_PASSES = 2   # 方向直方图平滑次数
SIFT_ORI_PEAK_RATIO = 0.8    # 产生新特征的方向峰值比率

SIFT_DESCR_SCL_FCTR = 3.0    # 描述子方向直方图大小因子
SIFT_DESCR_MAG_THR = 0.2     # 描述子向量元素幅度阈值
SIFT_INT_DESCR_FCTR = 512.0  # 浮点描述子转unsigned char的因子


class SIFTKeypoint:
    """SIFT关键点数据结构"""
    def __init__(self):
        self.x = 0.0          # x坐标（亚像素精度）
        self.y = 0.0          # y坐标（亚像素精度）
        self.octave = 0       # 所在组
        self.layer = 0        # 组内层
        self.scale = 0.0      # 尺度
        self.ori = 0.0        # 方向（弧度）
        self.descriptor = None  # 128维描述子
        
        # 检测数据（用于调试和理解）
        self.r = 0            # DoG空间中的行（整数）
        self.c = 0            # DoG空间中的列（整数）
        self.sub_interval = 0.0  # 亚层偏移
        

class DetailedSIFT:
    """详细的SIFT实现，展示每个步骤"""
    
    def __init__(self):
        self.octaves = []           # 高斯金字塔（尺度空间）
        self.dog_pyramids = []      # DoG金字塔
        self.keypoints = []         # 检测到的关键点
        
    def compute(self, image: np.ndarray) -> List[SIFTKeypoint]:
        """
        完整的SIFT特征检测和描述流程
        
        步骤：
        1. 构建尺度空间（高斯金字塔）
        2. 构建DoG金字塔
        3. 尺度空间极值检测
        4. 关键点精确定位和边缘响应去除
        5. 方向分配
        6. 生成描述子
        """
        print("\n========== SIFT 详细处理流程 ==========\n")
        
        # 步骤1: 预处理 - 转换为灰度图并归一化
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        gray = gray.astype(np.float32)
        
        print(f"步骤1: 图像预处理")
        print(f"  - 输入图像尺寸: {image.shape}")
        print(f"  - 灰度图尺寸: {gray.shape}")
        
        # 步骤2: 构建高斯金字塔（尺度空间）
        print(f"\n步骤2: 构建高斯金字塔（尺度空间）")
        self.build_gaussian_pyramid(gray)
        
        # 步骤3: 构建DoG金字塔
        print(f"\n步骤3: 构建DoG（Difference of Gaussian）金字塔")
        self.build_dog_pyramid()
        
        # 步骤4: 尺度空间极值检测
        print(f"\n步骤4: 尺度空间极值检测")
        raw_keypoints = self.detect_keypoints()
        print(f"  - 初步检测到 {len(raw_keypoints)} 个候选关键点")
        
        # 步骤5: 关键点精确定位和过滤
        print(f"\n步骤5: 关键点精确定位和过滤")
        refined_keypoints = self.refine_keypoints(raw_keypoints)
        print(f"  - 精确定位后保留 {len(refined_keypoints)} 个关键点")
        
        # 步骤6: 方向分配
        print(f"\n步骤6: 关键点方向分配")
        oriented_keypoints = self.assign_orientations(refined_keypoints)
        print(f"  - 方向分配后得到 {len(oriented_keypoints)} 个关键点（含多方向）")
        
        # 步骤7: 生成描述子
        print(f"\n步骤7: 生成128维SIFT描述子")
        self.compute_descriptors(oriented_keypoints)
        print(f"  - 最终生成 {len(oriented_keypoints)} 个SIFT特征")
        
        self.keypoints = oriented_keypoints
        print(f"\n========== SIFT 处理完成 ==========\n")
        
        return self.keypoints
    
    def build_gaussian_pyramid(self, image: np.ndarray):
        """
        构建高斯金字塔（尺度空间）
        
        理论：
        - 每组(octave)包含 s+3 层，其中s=SIFT_INTVLS (通常为3)
        - 每组之间通过2倍下采样连接
        - 组内通过不同sigma的高斯模糊产生不同尺度
        """
        # 第一步：将图像上采样2倍（Lowe论文中的做法，可以检测更多特征）
        base = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), 
                         interpolation=cv2.INTER_LINEAR)
        
        # 计算需要应用的初始模糊
        # 假设输入图像已有 SIFT_INIT_SIGMA=0.5 的模糊
        # 上采样后变为 0.5*2=1.0
        # 我们需要达到 SIFT_SIGMA=1.6
        sig_diff = np.sqrt(max(SIFT_SIGMA**2 - (2*SIFT_INIT_SIGMA)**2, 0.01))
        base = cv2.GaussianBlur(base, (0, 0), sig_diff)
        
        # 计算组数：确保最小的图像至少有几个像素
        num_octaves = int(np.log2(min(base.shape[0], base.shape[1]))) - 2
        print(f"  - 金字塔组数: {num_octaves}")
        print(f"  - 每组层数: {SIFT_INTVLS + 3}")
        
        # k是尺度空间中相邻层之间的倍数
        k = 2 ** (1.0 / SIFT_INTVLS)
        print(f"  - 层间尺度因子 k = 2^(1/{SIFT_INTVLS}) = {k:.4f}")
        
        self.octaves = []
        
        for o in range(num_octaves):
            octave = []
            print(f"  - 构建第 {o} 组，图像尺寸: {base.shape}")
            
            # 每组有 s+3 层
            for s in range(SIFT_INTVLS + 3):
                if s == 0:
                    # 第0层就是base（已经模糊到sigma）
                    sigma = SIFT_SIGMA
                    octave.append(base.copy())
                else:
                    # 后续层在前一层基础上进一步模糊
                    sigma = SIFT_SIGMA * (k ** s)
                    # 计算相对于前一层需要增加的模糊
                    sigma_prev = SIFT_SIGMA * (k ** (s-1))
                    sigma_diff = np.sqrt(sigma**2 - sigma_prev**2)
                    
                    blurred = cv2.GaussianBlur(octave[-1], (0, 0), sigma_diff)
                    octave.append(blurred)
                
                if s < 3:  # 只打印前几层的信息
                    print(f"    层 {s}: sigma = {sigma:.3f}")
            
            self.octaves.append(octave)
            
            # 准备下一组的base图像：取当前组倒数第3层，然后下采样
            if o < num_octaves - 1:
                base = cv2.resize(octave[-3], 
                                (octave[-3].shape[1]//2, octave[-3].shape[0]//2),
                                interpolation=cv2.INTER_NEAREST)
    
    def build_dog_pyramid(self):
        """
        构建DoG（Difference of Gaussian）金字塔
        
        理论：
        - DoG = G(σ₁) - G(σ₂)，近似于尺度归一化的LoG（Laplacian of Gaussian）
        - 用于检测尺度空间中的blob特征
        """
        self.dog_pyramids = []
        
        for o, octave in enumerate(self.octaves):
            dog_octave = []
            print(f"  - 第 {o} 组: 生成 {len(octave)-1} 层DoG")
            
            for i in range(len(octave) - 1):
                dog = octave[i+1] - octave[i]
                dog_octave.append(dog)
            
            self.dog_pyramids.append(dog_octave)
    
    def detect_keypoints(self) -> List[SIFTKeypoint]:
        """
        尺度空间极值检测
        
        理论：
        - 在DoG金字塔中，比较每个点与其26个邻域（同层8个+上下层各9个）
        - 如果是极值点（最大或最小），则作为候选关键点
        """
        candidates = []
        
        for octave_idx, dog_octave in enumerate(self.dog_pyramids):
            # 只检测中间层（第1层到倒数第2层）
            for layer_idx in range(1, len(dog_octave) - 1):
                dog_curr = dog_octave[layer_idx]
                dog_prev = dog_octave[layer_idx - 1]
                dog_next = dog_octave[layer_idx + 1]
                
                rows, cols = dog_curr.shape
                count = 0
                
                # 遍历图像（避开边界）
                for r in range(SIFT_IMG_BORDER, rows - SIFT_IMG_BORDER):
                    for c in range(SIFT_IMG_BORDER, cols - SIFT_IMG_BORDER):
                        val = dog_curr[r, c]
                        
                        # 检查是否为极值
                        if self.is_extremum(dog_prev, dog_curr, dog_next, r, c):
                            kp = SIFTKeypoint()
                            kp.octave = octave_idx
                            kp.layer = layer_idx
                            kp.r = r
                            kp.c = c
                            candidates.append(kp)
                            count += 1
                
                if count > 0:
                    print(f"    第 {octave_idx} 组第 {layer_idx} 层: 检测到 {count} 个极值点")
        
        return candidates
    
    def is_extremum(self, prev_layer, curr_layer, next_layer, r, c) -> bool:
        """
        判断点(r,c)在curr_layer是否为3x3x3邻域内的极值
        """
        val = curr_layer[r, c]
        
        # 对比度阈值预筛选
        if abs(val) <= 0.5 * SIFT_CONTR_THR / SIFT_INTVLS:
            return False
        
        # 检查是否为最大值或最小值
        is_max = True
        is_min = True
        
        # 检查当前层的8邻域
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                neighbor = curr_layer[r + dr, c + dc]
                if val <= neighbor:
                    is_max = False
                if val >= neighbor:
                    is_min = False
                if not is_max and not is_min:
                    return False
        
        # 检查上下层的9邻域
        for layer in [prev_layer, next_layer]:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    neighbor = layer[r + dr, c + dc]
                    if val <= neighbor:
                        is_max = False
                    if val >= neighbor:
                        is_min = False
                    if not is_max and not is_min:
                        return False
        
        return is_max or is_min
    
    def refine_keypoints(self, candidates: List[SIFTKeypoint]) -> List[SIFTKeypoint]:
        """
        关键点精确定位和过滤
        
        理论：
        1. 亚像素精确定位：使用泰勒展开式拟合3D二次函数
        2. 去除低对比度点：|D(x)| < SIFT_CONTR_THR
        3. 去除边缘响应：使用Hessian矩阵的迹和行列式
        """
        refined = []
        discarded_contrast = 0
        discarded_edge = 0
        discarded_interp = 0
        
        for kp in candidates:
            dog_octave = self.dog_pyramids[kp.octave]
            
            # 迭代精确定位
            r, c, layer = kp.r, kp.c, kp.layer
            
            for step in range(SIFT_MAX_INTERP_STEPS):
                # 获取当前位置的DoG值和导数
                curr_dog = dog_octave[layer]
                prev_dog = dog_octave[layer - 1]
                next_dog = dog_octave[layer + 1]
                
                # 计算梯度（一阶导数）
                dx = (curr_dog[r, c+1] - curr_dog[r, c-1]) / 2.0
                dy = (curr_dog[r+1, c] - curr_dog[r-1, c]) / 2.0
                ds = (next_dog[r, c] - prev_dog[r, c]) / 2.0
                grad = np.array([dx, dy, ds])
                
                # 计算Hessian矩阵（二阶导数）
                v = curr_dog[r, c]
                dxx = curr_dog[r, c+1] + curr_dog[r, c-1] - 2*v
                dyy = curr_dog[r+1, c] + curr_dog[r-1, c] - 2*v
                dss = next_dog[r, c] + prev_dog[r, c] - 2*v
                
                dxy = (curr_dog[r+1, c+1] - curr_dog[r+1, c-1] - 
                       curr_dog[r-1, c+1] + curr_dog[r-1, c-1]) / 4.0
                dxs = (next_dog[r, c+1] - next_dog[r, c-1] - 
                       prev_dog[r, c+1] + prev_dog[r, c-1]) / 4.0
                dys = (next_dog[r+1, c] - next_dog[r-1, c] - 
                       prev_dog[r+1, c] + prev_dog[r-1, c]) / 4.0
                
                H = np.array([[dxx, dxy, dxs],
                             [dxy, dyy, dys],
                             [dxs, dys, dss]])
                
                # 求解 H * offset = -grad
                try:
                    offset = -np.linalg.lstsq(H, grad, rcond=None)[0]
                except:
                    break
                
                # 如果偏移量很小，说明已经收敛
                if abs(offset[0]) < 0.5 and abs(offset[1]) < 0.5 and abs(offset[2]) < 0.5:
                    # 计算精确位置的DoG值
                    interp_val = v + 0.5 * np.dot(grad, offset)
                    
                    # 对比度检查
                    if abs(interp_val) < SIFT_CONTR_THR / SIFT_INTVLS:
                        discarded_contrast += 1
                        break
                    
                    # 边缘响应检查（使用2D Hessian）
                    trace = dxx + dyy
                    det = dxx * dyy - dxy * dxy
                    
                    if det <= 0 or trace*trace/det >= ((SIFT_CURV_THR+1)**2)/SIFT_CURV_THR:
                        discarded_edge += 1
                        break
                    
                    # 保存精确定位的关键点
                    kp.r = r
                    kp.c = c
                    kp.layer = layer
                    kp.sub_interval = offset[2]
                    
                    # 计算在原图中的坐标（考虑上采样和组的下采样）
                    scale_factor = 2 ** kp.octave
                    kp.x = (c + offset[0]) * scale_factor
                    kp.y = (r + offset[1]) * scale_factor
                    
                    # 计算实际尺度
                    k = 2 ** (1.0 / SIFT_INTVLS)
                    kp.scale = SIFT_SIGMA * (k ** (layer + offset[2])) * scale_factor
                    
                    refined.append(kp)
                    break
                
                # 更新位置
                c += int(round(offset[0]))
                r += int(round(offset[1]))
                layer += int(round(offset[2]))
                
                # 检查是否越界
                rows, cols = curr_dog.shape
                if (layer < 1 or layer >= len(dog_octave) - 1 or
                    r < SIFT_IMG_BORDER or r >= rows - SIFT_IMG_BORDER or
                    c < SIFT_IMG_BORDER or c >= cols - SIFT_IMG_BORDER):
                    discarded_interp += 1
                    break
            else:
                # 超过最大迭代次数
                discarded_interp += 1
        
        print(f"  - 过滤统计: 低对比度={discarded_contrast}, "
              f"边缘响应={discarded_edge}, 插值失败={discarded_interp}")
        
        return refined
    
    def assign_orientations(self, keypoints: List[SIFTKeypoint]) -> List[SIFTKeypoint]:
        """
        为关键点分配主方向
        
        理论：
        - 使用关键点邻域内的梯度方向和幅度构建方向直方图
        - 选择峰值方向作为主方向
        - 如果有其他接近峰值的方向，创建额外的关键点
        """
        oriented_kps = []
        
        for kp in keypoints:
            # 获取关键点所在的高斯图像
            octave = self.octaves[kp.octave]
            img = octave[kp.layer]
            
            scale_factor = 2 ** kp.octave
            r = int(round(kp.y / scale_factor))
            c = int(round(kp.x / scale_factor))
            
            # 计算方向直方图的权重sigma
            sigma = SIFT_ORI_SIG_FCTR * SIFT_SIGMA * (2 ** (kp.layer / SIFT_INTVLS))
            radius = int(round(SIFT_ORI_RADIUS * sigma))
            
            # 创建方向直方图
            hist = np.zeros(SIFT_ORI_HIST_BINS)
            
            rows, cols = img.shape
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    rr = r + i
                    cc = c + j
                    
                    if rr <= 0 or rr >= rows-1 or cc <= 0 or cc >= cols-1:
                        continue
                    
                    # 计算梯度
                    dx = img[rr, cc+1] - img[rr, cc-1]
                    dy = img[rr+1, cc] - img[rr-1, cc]
                    
                    mag = np.sqrt(dx*dx + dy*dy)
                    ori = np.arctan2(dy, dx)  # 弧度
                    
                    # 高斯加权
                    weight = np.exp(-(i*i + j*j) / (2*sigma*sigma))
                    
                    # 添加到直方图
                    bin_idx = int(round(SIFT_ORI_HIST_BINS * (ori + np.pi) / (2*np.pi))) % SIFT_ORI_HIST_BINS
                    hist[bin_idx] += weight * mag
            
            # 平滑直方图
            for _ in range(SIFT_ORI_SMOOTH_PASSES):
                hist = self.smooth_histogram(hist)
            
            # 找到最大峰值
            max_val = np.max(hist)
            
            # 寻找所有超过阈值的峰值
            for i in range(SIFT_ORI_HIST_BINS):
                # 检查是否为局部极大值
                prev_val = hist[(i-1) % SIFT_ORI_HIST_BINS]
                next_val = hist[(i+1) % SIFT_ORI_HIST_BINS]
                
                if hist[i] > prev_val and hist[i] > next_val and hist[i] >= SIFT_ORI_PEAK_RATIO * max_val:
                    # 抛物线插值得到精确的峰值位置
                    interp_peak = self.interpolate_peak(i, hist)
                    orientation = (2*np.pi * interp_peak / SIFT_ORI_HIST_BINS) - np.pi
                    
                    # 创建新的关键点（或使用原关键点）
                    new_kp = SIFTKeypoint()
                    new_kp.x = kp.x
                    new_kp.y = kp.y
                    new_kp.octave = kp.octave
                    new_kp.layer = kp.layer
                    new_kp.scale = kp.scale
                    new_kp.ori = orientation
                    new_kp.r = kp.r
                    new_kp.c = kp.c
                    new_kp.sub_interval = kp.sub_interval
                    
                    oriented_kps.append(new_kp)
        
        return oriented_kps
    
    def smooth_histogram(self, hist: np.ndarray) -> np.ndarray:
        """平滑直方图（循环卷积）"""
        n = len(hist)
        smoothed = np.zeros_like(hist)
        for i in range(n):
            smoothed[i] = 0.25 * hist[(i-1)%n] + 0.5 * hist[i] + 0.25 * hist[(i+1)%n]
        return smoothed
    
    def interpolate_peak(self, peak_idx: int, hist: np.ndarray) -> float:
        """使用抛物线插值找到精确的峰值位置"""
        n = len(hist)
        prev = hist[(peak_idx - 1) % n]
        curr = hist[peak_idx]
        next_val = hist[(peak_idx + 1) % n]
        
        # 抛物线插值
        interp = peak_idx + 0.5 * (prev - next_val) / (prev - 2*curr + next_val)
        
        if interp < 0:
            interp += n
        elif interp >= n:
            interp -= n
        
        return interp
    
    def compute_descriptors(self, keypoints: List[SIFTKeypoint]):
        """
        计算SIFT描述子
        
        理论：
        - 在关键点周围取16x16的窗口
        - 分成4x4个子区域
        - 每个子区域计算8方向的梯度直方图
        - 最终得到4x4x8=128维描述子
        """
        for kp in keypoints:
            octave = self.octaves[kp.octave]
            img = octave[kp.layer]
            
            scale_factor = 2 ** kp.octave
            r = int(round(kp.y / scale_factor))
            c = int(round(kp.x / scale_factor))
            
            # 描述子窗口的大小
            hist_width = SIFT_DESCR_SCL_FCTR * SIFT_SIGMA * (2 ** (kp.layer / SIFT_INTVLS))
            radius = int(round(hist_width * np.sqrt(2) * (SIFT_DESCR_WIDTH + 1) / 2.0))
            
            cos_t = np.cos(-kp.ori)
            sin_t = np.sin(-kp.ori)
            
            # 初始化描述子数组 4x4x8
            descriptor = np.zeros((SIFT_DESCR_WIDTH, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS))
            
            rows, cols = img.shape
            
            # 遍历窗口内的像素
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    # 旋转坐标到关键点方向
                    rot_i = j * sin_t + i * cos_t
                    rot_j = j * cos_t - i * sin_t
                    
                    # 归一化到描述子坐标
                    bin_i = rot_i / hist_width + SIFT_DESCR_WIDTH / 2.0 - 0.5
                    bin_j = rot_j / hist_width + SIFT_DESCR_WIDTH / 2.0 - 0.5
                    
                    # 检查是否在描述子范围内
                    if bin_i > -1 and bin_i < SIFT_DESCR_WIDTH and \
                       bin_j > -1 and bin_j < SIFT_DESCR_WIDTH:
                        
                        rr = r + i
                        cc = c + j
                        
                        if rr > 0 and rr < rows-1 and cc > 0 and cc < cols-1:
                            # 计算梯度
                            dx = img[rr, cc+1] - img[rr, cc-1]
                            dy = img[rr+1, cc] - img[rr-1, cc]
                            
                            mag = np.sqrt(dx*dx + dy*dy)
                            ori = np.arctan2(dy, dx)
                            
                            # 相对于关键点方向
                            ori -= kp.ori
                            while ori < 0:
                                ori += 2*np.pi
                            while ori >= 2*np.pi:
                                ori -= 2*np.pi
                            
                            # 高斯加权
                            weight = np.exp(-(rot_i*rot_i + rot_j*rot_j) / 
                                          (2 * (0.5*SIFT_DESCR_WIDTH)**2))
                            
                            # 三线性插值到描述子
                            self.trilinear_interp(descriptor, bin_i, bin_j, ori, 
                                                weight * mag)
            
            # 展平为128维向量
            desc_vec = descriptor.flatten()
            
            # 归一化
            norm = np.linalg.norm(desc_vec)
            if norm > 0:
                desc_vec /= norm
            
            # 截断大值并重新归一化（增加鲁棒性）
            desc_vec = np.minimum(desc_vec, SIFT_DESCR_MAG_THR)
            norm = np.linalg.norm(desc_vec)
            if norm > 0:
                desc_vec /= norm
            
            # 转换为整数（0-255）
            desc_vec = np.round(desc_vec * SIFT_INT_DESCR_FCTR)
            desc_vec = np.clip(desc_vec, 0, 255).astype(np.uint8)
            
            kp.descriptor = desc_vec
    
    def trilinear_interp(self, descriptor, bin_i, bin_j, ori, weight):
        """三线性插值：将梯度信息分配到相邻的bin"""
        bin_o = ori * SIFT_DESCR_HIST_BINS / (2*np.pi)
        
        # 获取整数部分和小数部分
        i0 = int(np.floor(bin_i))
        j0 = int(np.floor(bin_j))
        o0 = int(np.floor(bin_o)) % SIFT_DESCR_HIST_BINS
        
        di = bin_i - i0
        dj = bin_j - j0
        do = bin_o - int(np.floor(bin_o))
        
        # 遍历相邻的8个bin（2x2x2）
        for i in range(2):
            ii = i0 + i
            if ii < 0 or ii >= SIFT_DESCR_WIDTH:
                continue
            wi = (1-di) if i == 0 else di
            
            for j in range(2):
                jj = j0 + j
                if jj < 0 or jj >= SIFT_DESCR_WIDTH:
                    continue
                wj = (1-dj) if j == 0 else dj
                
                for o in range(2):
                    oo = (o0 + o) % SIFT_DESCR_HIST_BINS
                    wo = (1-do) if o == 0 else do
                    
                    descriptor[ii, jj, oo] += weight * wi * wj * wo


def match_features_with_kdtree(kp1_list: List[SIFTKeypoint], 
                               kp2_list: List[SIFTKeypoint],
                               ratio_thresh: float = 0.7) -> List[Tuple[int, int]]:
    """
    使用KD-Tree进行特征匹配
    
    理论：
    - 构建KD-Tree加速最近邻搜索
    - 使用Lowe's ratio test：d1 < ratio * d2
    """
    print(f"\n========== 特征匹配（KD-Tree + Ratio Test）==========\n")
    
    # 提取描述子
    desc1 = np.array([kp.descriptor for kp in kp1_list], dtype=np.float32)
    desc2 = np.array([kp.descriptor for kp in kp2_list], dtype=np.float32)
    
    print(f"  - 图像1特征数: {len(kp1_list)}")
    print(f"  - 图像2特征数: {len(kp2_list)}")
    print(f"  - 比率阈值: {ratio_thresh}")
    
    # 使用OpenCV的FLANN进行KD-Tree匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=200)  # 对应C++中的KDTREE_BBF_MAX_NN_CHKS
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append((m.queryIdx, m.trainIdx))
    
    print(f"  - 匹配对数: {len(good_matches)}")
    print(f"\n========== 匹配完成 ==========\n")
    
    return good_matches


def visualize_matches(img1: np.ndarray, img2: np.ndarray,
                     kp1_list: List[SIFTKeypoint], 
                     kp2_list: List[SIFTKeypoint],
                     matches: List[Tuple[int, int]]) -> np.ndarray:
    """可视化匹配结果"""
    # 垂直拼接图像
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    width = max(w1, w2)
    stacked = np.zeros((h1 + h2, width, 3), dtype=np.uint8)
    
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1.copy()
    
    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2.copy()
    
    stacked[:h1, :w1] = img1_color
    stacked[h1:h1+h2, :w2] = img2_color
    
    # 绘制匹配线
    for idx1, idx2 in matches:
        kp1 = kp1_list[idx1]
        kp2 = kp2_list[idx2]
        
        pt1 = (int(round(kp1.x)), int(round(kp1.y)))
        pt2 = (int(round(kp2.x)), int(round(kp2.y)) + h1)
        
        cv2.line(stacked, pt1, pt2, (255, 0, 255), 1, cv2.LINE_AA)
    
    return stacked


def main():
    default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="详细的SIFT算法实现")
    parser.add_argument("--img1", default=os.path.join(default_dir, "sse1.bmp"))
    parser.add_argument("--img2", default=os.path.join(default_dir, "sse2.bmp"))
    parser.add_argument("--ratio", type=float, default=0.7, help="Lowe ratio threshold")
    args = parser.parse_args()
    
    # 加载图像
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    
    if img1 is None or img2 is None:
        print(f"错误：无法加载图像")
        return 1
    
    # 对两幅图像分别进行SIFT特征检测
    print(f"\n{'='*60}")
    print(f"处理图像1: {os.path.basename(args.img1)}")
    print(f"{'='*60}")
    sift1 = DetailedSIFT()
    kp1_list = sift1.compute(img1)
    
    print(f"\n{'='*60}")
    print(f"处理图像2: {os.path.basename(args.img2)}")
    print(f"{'='*60}")
    sift2 = DetailedSIFT()
    kp2_list = sift2.compute(img2)
    
    # 特征匹配
    matches = match_features_with_kdtree(kp1_list, kp2_list, args.ratio)
    
    # 可视化
    result = visualize_matches(img1, img2, kp1_list, kp2_list, matches)
    
    cv2.imshow("Detailed SIFT Matches", result)
    print(f"\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
