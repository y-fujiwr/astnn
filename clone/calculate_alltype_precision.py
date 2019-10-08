import pandas as pd
from pathlib import Path
import re
import sys
from sklearn.metrics import precision_recall_fscore_support,roc_curve,roc_auc_score
import os
import matplotlib.pyplot as plt

threshold = 0.15

def read_csv(filename):
    dataframe = pd.read_csv(filename)
    def extractNum(func):
        return int(re.sub("\\D", "", func))
    def extractScore(func):
        return float(func[1:-1])
    dataframe["trues"] = dataframe["trues"].apply(extractNum)
    dataframe["predicts"] = dataframe["predicts"].apply(extractNum)
    dataframe["scores"] = dataframe["scores"].apply(extractScore)
    return dataframe

target_directory = sys.argv[1]
file_list = list(Path(target_directory).glob("**/*.csv"))

data = read_csv(file_list[3])
for i in range(1,len(file_list)):
    dataForAppend = read_csv(file_list[i])
    dataForAppend = dataForAppend[dataForAppend["trues"]==1]
    data = data.append(dataForAppend)
TruePositive = len(data[(data["trues"]==1) & (data["scores"]>=threshold)])
FalsePositive = len(data[(data["trues"]==0) & (data["scores"]>=threshold)])
FalseNegative = len(data[(data["trues"]==1) & (data["scores"]<threshold)])
TrueNegative = len(data[(data["trues"]==0) & (data["scores"]<threshold)])
precision = TruePositive / (TruePositive+FalsePositive)
recall = TruePositive / (TruePositive+FalseNegative)
print("p:{},r:{},f:{}".format(precision,recall,2*precision*recall/(precision+recall)))

trues = data["trues"].values
scores = data["scores"].values

fpr,tpr,thresholds = roc_curve(trues,scores)
print("ROC_score: {}".format(roc_auc_score(trues,scores)))
plt.plot(fpr, tpr, marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.show()