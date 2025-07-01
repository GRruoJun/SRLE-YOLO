import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 生成一个100x100的随机特征图
H, W = 100, 100
feature_map = np.random.rand(H, W)

# 设置块的大小
block_size_h = H // 10
block_size_w = W // 10

# 创建图形并显示特征图
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(feature_map, cmap='viridis', interpolation='nearest')

# 绘制块的边界
for i in range(10):
    for j in range(10):
        rect = patches.Rectangle((j * block_size_w, i * block_size_h), block_size_w, block_size_h,
                                 linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
        ax.add_patch(rect)

# 设置标题和图形参数
ax.set_title("Feature Map with 10x10 Blocks")
ax.set_xticks([])  # 不显示x轴刻度
ax.set_yticks([])  # 不显示y轴刻度

# 显示图像
plt.show()
