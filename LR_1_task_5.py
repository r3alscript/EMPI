import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, 
    precision_score, f1_score, roc_curve, roc_auc_score
)

# Завантаження набору даних
df = pd.read_csv('data_metrics.csv')
df.head()

# Встановлюємо поріг 
thresh = 0.5
# Стовпці, які перетворять ймовірності на передбачені мітки
df['predicted_RF'] = (df.model_RF >= thresh).astype('int')
df['predicted_LR'] = (df.model_LR >= thresh).astype('int')

# Написання власних функцій для матриці помилок
# True Positive
def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

# False Negative
def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

# False Positive
def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

# True Negative
def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

def find_conf_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def ruzhynski_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

# Перевірка правильності власної матриці помилок
assert np.array_equal(ruzhynski_confusion_matrix(df.actual_label.values, df.predicted_RF.values), 
                      confusion_matrix(df.actual_label.values, df.predicted_RF.values)), 'Matrix RF failed'
assert np.array_equal(ruzhynski_confusion_matrix(df.actual_label.values, df.predicted_LR.values), 
                      confusion_matrix(df.actual_label.values, df.predicted_LR.values)), 'Matrix LR failed'


# Власні функції для метрик якості
def ruzhynski_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def ruzhynski_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)

def ruzhynski_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)

def ruzhynski_f1_score(y_true, y_pred):
    recall = ruzhynski_recall_score(y_true, y_pred)
    precision = ruzhynski_precision_score(y_true, y_pred)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# Перевірка метрик через assert
assert ruzhynski_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values, df.predicted_RF.values)
assert ruzhynski_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values, df.predicted_RF.values)
assert ruzhynski_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(df.actual_label.values, df.predicted_RF.values)
assert ruzhynski_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(df.actual_label.values, df.predicted_RF.values)

print('\n--- Оцінки з порогом = 0.5 ---')
print('Accuracy RF: %.3f' % ruzhynski_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print('Recall RF: %.3f' % ruzhynski_recall_score(df.actual_label.values, df.predicted_RF.values))
print('Precision RF: %.3f' % ruzhynski_precision_score(df.actual_label.values, df.predicted_RF.values))
print('F1 RF: %.3f' % ruzhynski_f1_score(df.actual_label.values, df.predicted_RF.values))

print('\n--- Оцінки з порогом = 0.25 (для RF) ---')
pred_rf_025 = (df.model_RF >= 0.25).astype('int')
print('Accuracy RF: %.3f' % ruzhynski_accuracy_score(df.actual_label.values, pred_rf_025))
print('Recall RF: %.3f' % ruzhynski_recall_score(df.actual_label.values, pred_rf_025))
print('Precision RF: %.3f' % ruzhynski_precision_score(df.actual_label.values, pred_rf_025))
print('F1 RF: %.3f' % ruzhynski_f1_score(df.actual_label.values, pred_rf_025))

# 4. Побудова ROC-кривих та розрахунок AUC
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)

print('\n--- ROC-AUC результати ---')
print('AUC RF: %.3f' % auc_RF)
print('AUC LR: %.3f' % auc_LR)

# Побудова графіка ROC-кривої
plt.figure(figsize=(8, 6))
plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='Random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='Perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()