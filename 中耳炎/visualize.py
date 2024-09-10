import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import os


def draw_roc(y_true, y_score, title_name, save_path):
    """

    :param y_true:实际类别标签
    :param y_score:模型对每个样本的预测分数
    :param title_name:
    :return:
    """
    fpr, tpr, threshold = roc_curve(y_true, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    # print('AUC:{}'.format(roc_auc))
    plt.figure(figsize=(7, 7), dpi=300)
    # plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='lime', linewidth=2,
             label='AUC = %0.4f' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    if os.path.exists(os.path.join(save_path, 'roc')) == False:
        os.makedirs(os.path.join(save_path, 'roc'))
    plt.savefig(os.path.join(save_path, "roc/{}_roc.jpg".format(title_name)))
    # print("{}_roc.jpg 已保存".format(title_name))
    plt.close()


def draw_cm(y_true, y_predict, title_name, labels_name, save_path):
    # print(title_name, labels_name)
    plt.rcParams['font.size'] = max(7, int(40 / len(labels_name)))
    cm = confusion_matrix(y_true, y_predict)
    norm_cm = np.zeros_like(cm, dtype=float)
    total = []
    for j in range(len(cm)):
        now_total = 0
        for i in range(len(cm)):
            now_total += cm[j, i]
        total.append(now_total)
    for i in range(len(cm)):
        for j in range(len(cm)):
            norm_cm[j, i] = cm[j, i] / total[j]
    plt.figure(figsize=(10, 7), dpi=300)
    plt.imshow(norm_cm, interpolation='nearest', cmap="Blues")  # 在特定的窗口上显示图像

    for i in range(len(norm_cm)):
        for j in range(len(norm_cm)):
            if norm_cm[j, i] >= 0.5:
                c = 'white'
            else:
                c = 'black'
            plt.annotate("{}\n({:.0%})".format(cm[j, i], norm_cm[j, i]), xy=(i, j), horizontalalignment='center',
                         verticalalignment='center', color=c)
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    # plt.title(title)    # 图像标题
    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')
    if not os.path.exists(os.path.join(save_path, 'cm')):
        os.makedirs(os.path.join(save_path, 'cm'))
    plt.savefig(os.path.join(save_path, "cm/{}_cm.jpg".format(title_name)))
    print("{}_cm.jpg 已保存".format(title_name))
    plt.close()
    return cm
