from ultralytics import YOLO

#model = YOLO("yolov8n.pt")
model = YOLO("")  # 模型文件路径

results = model("/dataspace/grj/1.jpg", visualize=True)  # 要预测图片路径和使用可视化
