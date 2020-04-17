import pandas as pd
from pathlib import Path
import re
import sys
from sklearn.metrics import precision_recall_fscore_support,roc_curve,roc_auc_score
import os
import matplotlib.pyplot as plt
import numpy as np
import javalang

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
data = read_csv(file_list[0])

for i in range(1,len(file_list)):
    dataForAppend = read_csv(file_list[i])
    dataForAppend = dataForAppend[dataForAppend["trues"]==1]
    data = data.append(dataForAppend)

if len(sys.argv) > 2:
    method_table = pd.read_csv(sys.argv[2])
    limit = 5
    methodlist = data["id1"].append(data["id2"]).drop_duplicates()
    shortmethod = []
    for i in methodlist:
        tokens = list(javalang.tokenizer.tokenize(method_table[method_table["id"]==i]["code"].iloc[0]))
        lines = 0
        for t in tokens:
            if t.value == ";":
                lines += 1
        if lines <= limit:
            shortmethod.append(i)
    print(f"The number of small methods is {len(shortmethod)}")
    data = data[(~data["id1"].isin(shortmethod)) & (~data["id2"].isin(shortmethod))]

for threshold in np.arange(0.05, 0.95, 0.05):
    TruePositive = len(data[(data["trues"]==1) & (data["scores"]>=threshold)])
    FalsePositive = len(data[(data["trues"]==0) & (data["scores"]>=threshold)])
    FalseNegative = len(data[(data["trues"]==1) & (data["scores"]<threshold)])
    TrueNegative = len(data[(data["trues"]==0) & (data["scores"]<threshold)])
    try:
        precision = TruePositive / (TruePositive+FalsePositive)
        recall = TruePositive / (TruePositive+FalseNegative)
    except ZeroDivisionError:
        precision = 0
        recall = 1
    print(f"threshold: {threshold}")
    print(f"clone-pairs:{TruePositive+FalseNegative}, non-clone-pairs:{FalsePositive+TrueNegative}")
    print("p:{},r:{},f:{}".format(precision,recall,2*precision*recall/(precision+recall)))
    #print(TruePositive / (TruePositive+FalseNegative))
    #print(FalsePositive / (TrueNegative+FalsePositive))

trues = data["trues"].values
scores = data["scores"].values

fpr,tpr,thresholds = roc_curve(trues,scores)
print("AUC_score: {}".format(roc_auc_score(trues,scores)))
exit()
plt.plot(fpr, tpr, marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.savefig(f"{target_directory}roccurve_.eps")