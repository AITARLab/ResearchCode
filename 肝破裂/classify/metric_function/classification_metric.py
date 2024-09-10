import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, confusion_matrix


class BinaryClassficationMetrics(object):
    def __init__(self, true_labels, predicted_labels, label_scores, n_classes=2):
        self.n_classes = n_classes
        self.cm = confusion_matrix(true_labels, predicted_labels, labels=range(self.n_classes))
        # print(self.cm.ravel())
        self.TN, self.FP, self.FN, self.TP = self.cm.ravel()
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.label_scores = label_scores
        # print(self.FP, self.FN, self.TP, self.TN)

    def calculate_accuracy(self):
        return accuracy_score(self.true_labels, self.predicted_labels)

    def calculate_precision(self):
        return precision_score(self.true_labels, self.predicted_labels)

    def calculate_f1_score(self):
        return f1_score(self.true_labels, self.predicted_labels)

    def calculate_AUC_CI(self, alpha=0.05, confidence=0.95):
        AUC = roc_auc_score(self.true_labels, self.label_scores)
        label = np.array(self.true_labels)
        n1, n2 = np.sum(label == 1), np.sum(label == 0)
        q1 = AUC / (2 - AUC)
        q2 = (2 * AUC ** 2) / (1 + AUC)
        se = np.sqrt((AUC * (1 - AUC) + (n1 - 1) * (q1 - AUC ** 2) + (n2 - 1) * (q2 - AUC ** 2)) / (n1 * n2))
        confidence_level = 1 - alpha
        z_lower, z_upper = scipy.stats.norm.interval(confidence_level)
        lower, upper = AUC + z_lower * se, AUC + z_upper * se
        return AUC, lower, upper

    def calculate_sensitivity(self):
        if self.TP + self.FN == 0:
            return 0
        else:
            return self.TP / (self.TP + self.FN)

    def calculate_specificity(self):
        if self.TN + self.FP == 0:
            return 0
        else:
            return self.TN / (self.TN + self.FP)


