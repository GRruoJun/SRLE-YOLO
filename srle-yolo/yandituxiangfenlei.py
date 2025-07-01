# -*- coding: utf-8 -*-
from ultralytics import YOLO
import matplotlib
import os
# matplotlib.use('TkAgg')
from sklearn.metrics import accuracy_score, precision_score,f1_score,recall_score
from sklearn.metrics import classification_report

# 验证结果
model = YOLO('/dataspace/grj/laov8/runs/train/exp241/weights/best.pt')
# 模型标签
names = {0: 'A', 1: 'C', 2: 'D', 3: 'G', 4: 'H',5:'M',6:'N',7:'O',}
# 验证集路径
base_path = '/dataspace/grj/yandituxxiang/Training_Dataset/test'

# 将names的key与value值互换，存入dict_names中
dict_names = {v: k for k, v in names.items()}
# 存储真实标签
real_labels = []
# 存储预测标签
pre_labels = []
# 遍历base_path下的所有文件夹，每个文件夹是一个分类
for i in os.listdir(base_path):
    label = dict_names[i]
    # 获取base_path下的所有文件夹下的所有图片
    for j in os.listdir(os.path.join(base_path, i)):
        # 获取图片的路径
        img_path = os.path.join(base_path, i, j)
        # 检测图片
        res = model.predict(img_path)[0]
        # 图片真实标签
        real_labels.append(label)
        # 图片预测标签
        pre_labels.append(res.probs.top1)
print("每个类别的精确率、召回率和F1-Score：")
print(classification_report(real_labels, pre_labels, target_names=list(names.values())))
# 计算并打印一系列评估指标，包括准确率、精确率、F1分数和召回率
# 参数:
# real_labels: 真实标签列表，表示样本的真实类别
# pre_labels: 预测标签列表，表示模型预测的样本类别
print('单独计算的准确率、精确率、F1分数和召回率:')
# 1. accuracy_score: 准确率，表示预测正确的样本占总样本的比例
print('accuracy_score:',accuracy_score(real_labels, pre_labels))
# 2. precision_score: 精确率，表示预测为正类且实际为正类的样本占预测为正类样本的比例
print('precision_score:',precision_score(real_labels, pre_labels, average='macro'))
# 3. f1_score: F1分数，是精确率和召回率的调和平均值，综合评估精确度和召回率
print('f1_score:',f1_score(real_labels, pre_labels, average='macro'))
# 4. recall_score: 召回率，表示预测为正类且实际为正类的样本占实际正类样本的比例
print('recall_score:',recall_score(real_labels, pre_labels, average='macro'))
