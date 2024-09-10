# -*- coding = utf-8 -*-
# @time:2024/7/22 19:22
# Author:lyh
# @File:val.py
# @Software:PyCharm
import numpy as np
from sklearn.preprocessing import label_binarize
from ultralytics import YOLO
import os
from visualize import *
import cv2
from PIL import Image
import pandas as pd
from scipy.stats import norm
import scipy.stats as stats


def show_cm_metric(cm):
    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]
    tn = cm[0, 0]
    info = ['TP:{} FN:{} FP:{} TN:{}'.format(tp, fn, fp, tn), 'precision: {:.4f}'.format(tp / (tp + fp)),
            'recall: {:.4f}'.format(tp / (tp + fn)),
            'F1: {:.4f}'.format(2 * (((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn))))),
            "sensitivity: {:.4f}".format(tp / (tp + fn)), "specificity: {:.4f}".format(tn / (tn + fp))]
    return info, [tp, fn, fp, tn, tp / (tp + fp), tp / (tp + fn),
            2 * (((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn)))),
            tp / (tp + fn), tn / (tn + fp)]


def calc_multi_class(cm, writer, classes=3):
    all_data = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(classes):
        new_cm = np.array([[0, 0], [0, 0]])
        for k1 in range(classes):
            if k1 == i:
                x = 1
            else:
                x = 0
            for k2 in range(classes):
                if k2 == i:
                    y = 1
                else:
                    y = 0
                new_cm[x, y] += cm[k1, k2]
        info, data = show_cm_metric(new_cm)
        all_data = all_data + np.array(data)
        writer.write("Class:{}={}\n".format(i, info))
    writer.write("Avg:{}\n".format(np.array(all_data) / 3))


def get_labels(label_file_fold):
    label_files = os.listdir(label_file_fold)
    labels = []
    for label_file in label_files:
        label_file = os.path.join(label_file_fold, label_file)
        with open(label_file, 'r', encoding='utf-8') as file:
            # 读取文件的第一行
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                label = line.split()[0]
                labels.append(label)
    return [int(item) for item in labels]


def bootstrap_auc(y, pred, classes, bootstraps=100, fold_size=1000):
    statistics = np.zeros((len(classes), bootstraps))
    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        # df.
        df.loc[:, 'y'] = y
        df.loc[:, 'pred'] = pred
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n=int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n=int(fold_size * (1 - prevalence)), replace=True)
            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    # 计算置信区间
    lower_bound = np.percentile(statistics, 2.5)
    upper_bound = np.percentile(statistics, 97.5)
    return lower_bound, upper_bound
    # alpha = 0.05
    # ci = stats.norm.interval(alpha, loc=statistics.mean(), scale=statistics.std() / np.sqrt(statistics))
    # print('Bootstrap Confidence Interval for Mean:', ci)
    # return ci[0], ci[1]
    # lower, upper = np.min(statistics, axis=1)[1], np.max(statistics, axis=1)[1]
    # return lower, upper

# label_file = f'balanced_test/labels'
# labels = get_labels(label_file)
# c = [0, 0, 0]
# for i in labels:
#     c[i] += 1
# print(c)
# exit(0)
plt.rcParams['font.size'] = max(7, int(40 / 3))
name = ['NME vs rest', 'OME vs rest', 'COM vs rest', 'Mean']
col = ['lime', 'orange', 'crimson', 'blue']
result_txt = open("./test_result/result-test.txt", "w")
for i in range(5):
    result_txt.write("Fold_{}:\n".format(i))
    # model_path = f'./runs/detect/epoch25-512-0.0005-16/custom_yolo_fold_{i}/weights/best.pt'
    # model_path = f'./runs/detect/epoch40-512-0.0005-16-new/custom_yolo_fold_{i}/weights/best.pt'
    model_path = f'./runs/detect/epoch50-512-0.0005-16-new/custom_yolo_fold_{i}/weights/best.pt'
    # 加载训练好的模型
    model = YOLO(model_path)
    pred_labels = []
    true_labels = []
    pred_scores = [[], [], [], []]
    no_detections = 0
    actual_labels = [[], [], [], []]
    # img_paths = os.listdir(f'dataset/fold{i}/images')
    # label_file = f'dataset/fold{i}/labels'
    img_paths = os.listdir(f'balanced_test/images')
    label_file = f'balanced_test/labels'
    # img_paths = os.listdir(f'dataset/test/images')
    # label_file = f'dataset/test/labels'
    labels = get_labels(label_file)
    # 遍历验证集图片并进行预测
    right = 0
    r = [0, 0, 0]
    c = [0, 0, 0]
    iindex = 0
    total = len(labels)
    for index, img_path in enumerate(img_paths):
        # img_path = os.path.join(f'dataset/fold{i}/images', img_path)
        img_all_path = os.path.join(f'balanced_test/images', img_path)
        # img_all_path = os.path.join(f'dataset/test/images', img_path)
        # 使用PIL打开图像
        pil_image = Image.open(img_all_path)
        # 将PIL图像转换为cv2图像（‌BGR格式）‌
        image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        # image = cv2.imread(img_all_path)
        (h, w) = image.shape[:2]
        result = model.predict(img_all_path, stream=True)
        for j in result:
            if j.boxes.cls.numel() > 0:
                # print('-------------------------------')
                # print(j.boxes.conf)
                pred_label = int(j.boxes.cls[0].item())
                pred_score = float(j.boxes.conf[0].item())
                pred_labels.append(pred_label)
                true_labels.append(labels[iindex])
                c[labels[iindex]] += 1
                for w in range(3):
                    if labels[iindex] == w:
                        actual_labels[w].append(1)
                    else:
                        actual_labels[w].append(0)
                if pred_label == labels[iindex]:
                    r[pred_label] += 1
                    pred_scores[3].append(pred_score)
                    actual_labels[3].append(1)
                    for w in range(3):
                        if w == labels[iindex]:
                            if pred_score < 0.5:
                                pred_scores[w].append(1 - pred_score)
                            else:
                                pred_scores[w].append(pred_score)
                        else:
                            if pred_score > 0.5:
                                pred_scores[w].append(1 - pred_score)
                            else:
                                pred_scores[w].append(pred_score)
                    right += 1
                else:
                    pred_scores[3].append(1 - pred_score)
                    actual_labels[3].append(0)
                    for w in range(3):
                        if w == labels[iindex]:
                            if pred_score > 0.5:
                                pred_scores[w].append(1 - pred_score)
                            else:
                                pred_scores[w].append(pred_score)
                        else:
                            if pred_label == w:
                                if pred_score < 0.5:
                                    pred_scores[w].append(1 - pred_score)
                                else:
                                    pred_scores[w].append(pred_score)
                            else:
                                if pred_score > 0.5:
                                    pred_scores[w].append(1 - pred_score)
                                else:
                                    pred_scores[w].append(pred_score)
                # actual_labels.append(labels[index])
            else:
                no_detections += 1
            iindex += 1
    # print(r, c)
    fpr = []
    tpr = []
    roc_auc = []
    result_txt.write("0:{}/{}:{:.4f}|1:{}/{}:{:.4f}|2:{}/{}:{:.4f}\n".format(
        r[0], c[0], r[0] / c[0], r[1], c[1], r[1] / c[1], r[2], c[2], r[2] / c[2]
    ))
    for w in range(4):
        f, t, _ = roc_curve(actual_labels[w], pred_scores[w])
        fpr.append(f)
        tpr.append(t)
        auc_v = auc(f, t)
        roc_auc.append(auc_v)
        l, r = bootstrap_auc(actual_labels[w], pred_scores[w], [0, 1])
        result_txt.write(f"AUC{w}: {auc_v}||95%CI:{l}-{r}\n")
        draw_roc(actual_labels[w], pred_scores[w], f"AUC-Fold{i}-OvR{w}", "./test_result")
    plt.figure(figsize=(7, 7), dpi=300)
    for w in range(4):
        plt.plot(fpr[w], tpr[w], color=col[w], linewidth=2, label='{}, AUC:{:.4f}'.format(name[w], roc_auc[w]))
    plt.plot([0, 1], [0, 1], color='black', linewidth=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc="lower right")
    plt.savefig("./test_result/Fold{}_all_roc.jpg".format(i))
    plt.close()
    result_txt.write("||{}/{}:{:.4f}||{}/{}:{:.4f}\n".format(no_detections, total, no_detections / total, right, total,
                                                             right / total))
    cm = draw_cm(true_labels, pred_labels, f'fold{i}', ['NME', 'OME', 'COM'], './test_result')
    calc_multi_class(cm, result_txt)

result_txt.write('\n')

result_txt.close()
# for j in result:
#     print(j.boxes)
# 解析YOLO输出
# for output in result:
#     if output.boxes.cls.numel() <= 0:
#         continue
#     for k in range(output.boxes.cls.numel()):
#         confidence = np.array(output.boxes.conf.cpu())[k]
#         classID = names[int(output.boxes.cls.cpu()[k])]
#         print(confidence, classID, output.boxes.xywhn[k])
#         box = np.array(output.boxes.xywhn[k].cpu()) * np.array([w, h, w, h])
#         print(box)
#         (centerX, centerY, width, height) = box.astype("int")
#         # 画出检测到的目标
#         x = int(centerX - (width / 2))
#         y = int(centerY - (height / 2))
#         cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
#         cv2.putText(image, "{}:{:.2f}".format(classID, confidence), (x, y - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# cv2.imwrite("./balanced_test/result/{}.jpg".format(img_path.split('.')[0]), image)
"""
cls: tensor([1.], device='cuda:0')
conf: tensor([0.4533], device='cuda:0')
data: tensor([[110.8032, 263.3642, 208.2217, 331.0086,   0.4533,   1.0000]], device='cuda:0')
id: None
is_track: False
orig_shape: (512, 512)
shape: torch.Size([1, 6])
xywh: tensor([[159.5125, 297.1864,  97.4186,  67.6444]], device='cuda:0')
xywhn: tensor([[0.3115, 0.5804, 0.1903, 0.1321]], device='cuda:0')
xyxy: tensor([[110.8032, 263.3642, 208.2217, 331.0086]], device='cuda:0')
xyxyn: tensor([[0.2164, 0.5144, 0.4067, 0.6465]], device='cuda:0')
Speed: 7.0ms preprocess, 16.5ms inference, 3.0ms postprocess per image at shape (1, 3, 512, 512)
进程已结束,退出代码0
"""
