import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/dataspace/grj/laov8/rtdetr-best.pt')
    model.val(data=r'/dataspace/grj/laov8/VisDrone.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              save_json=False, # 这个保存coco精度指标的开关
              project='runs/val',
              name='exp',
              )