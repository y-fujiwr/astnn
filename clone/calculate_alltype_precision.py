import pandas as pd
from pathlib import Path
import re
import sys
def read_csv(filename):
    dataframe = pd.read_csv(filename)
    def extractNum(func):
        return int(re.sub("\\D", "", func))
    def extractScore(func):
        return float(func[1:-2])
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
TruePositive = len(data[(data["trues"]==1) & (data["predicts"]==1)])
FalsePositive = len(data[(data["trues"]==0) & (data["predicts"]==1)])
FalseNegative = len(data[(data["trues"]==1) & (data["predicts"]==0)])
TrueNegative = len(data[(data["trues"]==0) & (data["predicts"]==0)])
precision = TruePositive / (TruePositive+FalsePositive)
recall = TruePositive / (TruePositive+FalseNegative)
print("p:{},r:{},f:{}".format(precision,recall,2*precision*recall/(precision+recall)))