import pandas as pd
import javalang
import random

limit = 50
smallmethod =[]
method_table = pd.read_csv("sesame_funcs_all.csv")
pairs = pd.read_csv("sesame_pairs_selfcheck.csv")

posi = pairs[pairs["label"]!=0]
nega = pairs[pairs["label"]==0]

while(len(posi)>len(nega)):
    a = random.randint(0,len(method_table)-1)
    b = random.randint(0,len(method_table)-1)
    if a==b:
        continue
    elif a > b:
        temp = a
        a = b
        b = temp
    if len(pairs[(pairs["id1"].isin([method_table.iloc[a]["id"]])) & (pairs["id2"].isin([method_table.iloc[b]["id"]]))]) == 0:
        nega = nega.append(pd.Series([method_table.iloc[a]["id"],method_table.iloc[b]["id"],0],index=["id1","id2","label"]),ignore_index=True)

print(len(posi))
print(len(nega))

posi.append(nega).to_pickle("sesame_pair_ids_selfcheck.pkl")