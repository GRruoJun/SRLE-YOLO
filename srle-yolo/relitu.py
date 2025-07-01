from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2

# 加载 YOLOv8 自定义/预训练模型
model = YOLO(r"/dataspace/grj/laov8/runs/train/exp220/weights/best.pt")  # YOLOv8 自定义/预训练模型

# 读取单张图片
im0 = cv2.imread(r"/dataspace/grj/laov8/KSHDUIbiSHIYAN/csmjcj4.jpg")
assert im0 is not None, "Error reading image file"  # 确保图片文件可以读取，否则报错

# 为热图设置感兴趣的类别（这里设置为类别0）
classes_for_heatmap = [0,1,2,3,4,5,6,7]  # 生成热图的目标类别

# 初始化热图对象
heatmap_obj = heatmap.Heatmap()
# 设置热图参数
heatmap_obj.set_args(colormap=cv2.COLORMAP_HOT,  # 使用 Cividis 颜色映射
                     imw=im0.shape[1],  # 设置热图的宽度，应与图片宽度一致
                     imh=im0.shape[0],  # 设置热图的高度，应与图片高度一致
                     view_img=False)  # 是否显示热图

# 使用模型跟踪图片中的目标，并生成热图
results = model.track(im0, persist=True, classes=classes_for_heatmap)
im0 = heatmap_obj.generate_heatmap(im0, tracks=results)

# # 显示处理后的图片
# cv2.imshow("Processed Image", im0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 如果需要，还可以将处理后的图片保存到文件
cv2.imwrite("/dataspace/grj/laov8/KSHDUIbiSHIYAN/220/4.jpg", im0)