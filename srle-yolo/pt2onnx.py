from ultralytics import YOLO
import onnx
# 加载训练好的模型
model = YOLO("/dataspace/grj/laov8/runs/train/exp28/weights/best.pt")
# 将模型转为onnx格式
success = model.export(format='onnx')
# 增加特征图维度信息
model_file = '/dataspace/grj/laov8/runs/train/exp28/weights/best.onnx'
    # 加载刚转换好的best.onnx文件
onnx_model = onnx.load(model_file)
    # 重新保存为best2.onnx文件
onnx.save(onnx.shape_inference.infer_shapes(onnx_model), '/dataspace/grj/laov8/runs/train/exp28/weights/best2.onnx')
