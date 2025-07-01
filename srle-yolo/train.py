import sys
sys.path.append("/dataspace/grj/laov8")
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('')

    #model.load('/dataspace/grj/laov8/yolov8m.pt') # 是否加载预训练权重

    model.train(data=r'/dataspace/grj/laov8/VisDrone.yaml',
                # 如果任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=False,
                imgsz=640,
                epochs=1,
                single_cls=False,  # 是否是单类别检测
                batch=4,
                close_mosaic=10,
                workers=0,
                device='1',
                optimizer='SGD', # using SGD
                #resume='runs/train/exp28/weights/last.pt', # 如过想续训就设置last.pt的地址
                amp=False,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                name='exp',
                )