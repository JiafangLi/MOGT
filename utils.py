import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  roc_curve, auc
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef,roc_auc_score


def calculate_confidence_interval(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    n = len(scores)
    z = 1.96  # 95% 置信水平的Z值
    bound = (z * std / np.sqrt(n))
    return bound



def get_gene_list(rename=False,disease = "SCZ"):
    filename = "data/"+disease + "/" + disease + ".csv"
    gene_list = pd.read_csv(filename)
    gene_list = gene_list[["gene_name","gene_id"]]
    return gene_list.rename(columns={'gene_name' : 'Gene Name'}) if rename else gene_list


def drawROC(y_true,y_pric,disease):
    fpr, tpr, th = roc_curve(y_true, y_pric, pos_label=1)
    finalauc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("img/ROC/"+disease+"ROC.png")
    return  finalauc


def multi_diseases_roc(sampling_methods,disease,folds,colors, save=True, dpin=100):
    aucs = []
    for (fold, y, colorname) in zip(folds,sampling_methods, colors):
        y_true = y["y_true"]
        y_pred = y["y_score"]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        # 绘制 POC 曲线
        aucs.append(auc)
        plt.plot(fpr, tpr, lw=2, label='Fold {} (AUC={:.4})'.format(fold, auc), color=colorname)
    mean_auc = np.mean(aucs)
    plt.plot([], [], ' ', label='Fold Mean = {:.4f}'.format(mean_auc))
    plt.plot([0, 1], [0, 1], '--', lw=2, color='grey')
    plt.axis('square')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title(disease, fontsize=15)
    plt.legend(loc='lower right', fontsize=10)
    plt.savefig("img/ROC/"+disease+"ROC.svg", format='svg')
    plt.close()
    return plt


def caculateThreshold(y_true,y_pred):
    max_mcc = -1
    best_threshold = 0
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    for threshold in thresholds:
        yy_pred = (y_pred >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, yy_pred)
        if mcc > max_mcc:
            max_mcc = mcc
            best_threshold = threshold
    return best_threshold


def multi_diseases_auprc(sampling_methods,diseases,  colors, save=True, dpin=100):
    for (disease, y, colorname) in zip(diseases,sampling_methods, colors):
        y_true = y["y_true"]
        y_pred = y["y_score"]
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
        # 绘制 POC 曲线
        plt.plot(recall, precision, lw=2, label='{} (Auprc={:.3f})'.format(disease, auc(recall,precision)), color=colorname)
        plt.plot([0,1], [1,0], '--', lw=2, color='grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall', fontsize=10)
        plt.ylabel('Precision', fontsize=10)
        plt.title('Precision-Recall', fontsize=15)
        plt.legend(loc='upper right', fontsize=10)
    plt.savefig("img/ROC/"+"auprc.png", dpi=dpin)
    return plt