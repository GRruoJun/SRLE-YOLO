import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/dataspace/grj/laov8/runs/train/exp27/weights/best.pt') # select your model.pt path
    model.track(source='People.mp4',
                imgsz=640,
                project='runs/track',
                name='exp',
                save=True
                )