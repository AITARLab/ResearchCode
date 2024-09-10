import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import pickle
from scipy.stats import norm
from Visualization import *
from log import logger_config


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



def train(model_name, data_file, task, fold=5):
    """

    :param model_name:
    :param data_file:
    :param fold:
    :return:
    """
    logger = logger_config(log_path=f"log/{task}_{model_name}.txt", logging_name=model_name)
    # 导入数据集，划分特征和标签
    df = pd.read_excel(data_file).drop([0])
    X = df.drop(["ID", "patient", "surgery"], axis=1)
    y = df["surgery"]
   
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

    # 特征缩放
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)
    # 主成分分析
    transfer2 = PCA(n_components=0.95)
    X_train = transfer2.fit_transform(X_train)
    X_test = transfer2.transform(X_test)

    if model_name == "SVC":  # 支持向量机
        model = SVC(probability=True)
        model = GridSearchCV(model, param_grid={'C': [10 ** i for i in range(-3, 4)], 'gamma': [10 ** i for i in range(-4, 2)]}, cv=fold)
    elif model_name == "RandomForest":  # 随机森林
        model = RandomForestClassifier()
        model = GridSearchCV(model, param_grid={'n_estimators': np.arange(100, 150, 10), 'max_depth': np.arange(10, 20, 1)},cv=fold)
    elif model_name == "KN":  # K邻近
        model = KNeighborsClassifier()
        model = GridSearchCV(model, param_grid=[
            {'weights': ['uniform'], 'n_neighbors': [i for i in range(1, 11)]},
            {'weights': ['distance'], 'n_neighbors': [i for i in range(1, 11)], 'p': [i for i in range(1, 6)]}], cv=fold)
    elif model_name == "GaussianNB":  # 高斯朴素贝叶斯
        model = GaussianNB()
        model = GridSearchCV(model, param_grid={'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}, cv=fold)
    elif model_name == "AdaBoost":  # AdaBoost分类器
        model = AdaBoostClassifier()
        model = GridSearchCV(model, param_grid={'n_estimators': np.arange(100, 200, 10), 'learning_rate': [0.1, 0.5, 1.0] }, cv=fold)
    elif model_name == "GradientBoost":  # 梯度提升分类器
        model = GradientBoostingClassifier()
        model = GridSearchCV(model, param_grid={'learning_rate': [0.1, 0.01, 0.001], 'n_estimators': np.arange(100, 150,10), 'max_depth': np.arange(1, 10, 1)}, cv=fold)
    elif model_name == "XGBoost":  # XGBoost分类器
        model = xgb.XGBClassifier()
        model = GridSearchCV(model, param_grid={'n_estimators': np.arange(100, 200, 10),
                            'learning_rate': [0.1, 0.01, 0.001],
                            'max_depth': np.arange(1, 10, 1)}, cv=fold)

    # 模型拟合
    model.fit(X_train, y_train)
    save_name = f"weights/{model_name}.pkl"
    with open(save_name, 'wb') as file:
        pickle.dump(model, file)

    # 测试集
    y_pred_proba = model.predict_proba(X_test)
    logger.info(f"y_pred_proba:{y_pred_proba}")
    y_pred_proba_quant = y_pred_proba[:, 1]
    logger.info(f"y_pred_proba_quant:{y_pred_proba_quant}")
    threshold = 0.4
    y_pred = (y_pred_proba_quant > threshold).astype(int)

    # 评价指标
    auc = roc_auc_score(y_test, y_pred_proba_quant)
    accuracy = accuracy_score(y_test, y_pred)
    ci_lower,ci_upper = bootstrap_auc(y_test, y_pred_proba_quant,[0,1])

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, y_pred)
    logger.info("auc:{:.4f}  acc:{:.4f}  CI:[{:.4f},{:.4f}]  sensitivity:{:.4f}  specificity:{:.4f}  f1:{:.4f}".format(auc, accuracy, ci_lower, ci_upper, sensitivity, specificity, f1))

    # 绘图
    title_name = f"results/{task}_{model_name}"
    draw_cm(y_test, y_pred, title_name, ["0", "1"])
    draw_roc(y_test, y_pred_proba_quant, title_name)




