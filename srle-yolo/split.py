from PIL import Image

# 打开图片
img = Image.open('/dataspace/grj/划分图片2/下采样图片4.jpg')
width, height = img.size

# 计算每个子图的尺寸
sub_width = width // 4
sub_height = height // 4

# 分割图片
for i in range(4):
    for j in range(4):
        left = j * sub_width
        upper = i * sub_height
        right = (j + 1) * sub_width
        lower = (i + 1) * sub_height
        sub_img = img.crop((left, upper, right, lower))
        sub_img.save(f'/dataspace/grj/划分图片2/sub_{i}_{j}.jpg')