import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score, f1_score


def bootstrap_auc(y, pred, classes, bootstraps = 100, fold_size = 20):
    statistics = np.zeros((len(classes), bootstraps))
    for c in range(len(classes)):
        df = pd.DataFrame({'y': y, 'pred': pred})
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    lower_bound = np.percentile(statistics, 2.5)
    upper_bound = np.percentile(statistics, 97.5)
    return lower_bound, upper_bound


def read_excel_to_dict(file_path, sheet_name):  
    df = pd.read_excel(file_path, sheet_name=sheet_name)  
    result_dict = {}
    for index, row in df.iterrows():  
        img_id = row.iloc[0].split("_")[0]
        # if len(row) == 4:
        if img_id not in result_dict.keys():
            result_dict[img_id] = []
        result_dict[img_id].append([row.iloc[-2], row.iloc[-1]])
    
    for key in result_dict.keys():
        value = result_dict[key]
        result_dict[key] = np.mean(value, axis=0)
    return result_dict


def read_label(file_path, sheet_name):  
    df = pd.read_excel(file_path, sheet_name=sheet_name)  
    result_dict = {}
    for index, row in df.iterrows():  
        img_id = row.iloc[0].split("_")[0]
        if img_id not in result_dict.keys():
            result_dict[img_id] = row.iloc[1]
    return result_dict
  

with open("data/map.json","r",encoding="utf-8") as f:
    name_dict = json.load(f)

xlsx_file = "data/multidata.xlsx"
sheet1 = "resnet18"
sheet2 = "densnet"
sheet3 = "GaussianNB"
sheet4 = "1"
label_dict = read_label(xlsx_file, sheet4)
DL = read_excel_to_dict(xlsx_file, sheet4)
print(DL)
ML = read_excel_to_dict(xlsx_file, sheet3)

y_pred_proba = []
y_test = []
for p_id in DL.keys():
    p_name = name_dict[p_id]
    pred = (DL[p_id] + ML[p_name])/2
    y_pred_proba.append(pred)
    y_test.append(label_dict[p_id])

y_pred_proba = np.array(y_pred_proba)
# print(y_pred_proba)
y_pred_proba_quant = y_pred_proba[:, 1]
threshold = 0.5
y_pred = (y_pred_proba_quant > threshold).astype(int)
auc = roc_auc_score(y_test, y_pred_proba_quant)
accuracy = accuracy_score(y_test, y_pred)
ci_lower,ci_upper = bootstrap_auc(y_test, y_pred_proba_quant,[0,1])

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
f1 = f1_score(y_test, y_pred)
print("auc:{:.4f}  acc:{:.4f}  CI:[{:.4f},{:.4f}]  sensitivity:{:.4f}  specificity:{:.4f}  f1:{:.4f}".format(auc, accuracy, ci_lower, ci_upper, sensitivity, specificity, f1))