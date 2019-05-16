from pathlib import Path
from tqdm import tqdm,trange
import os
import pandas as pd

CONS = 30000000

def makeFuncTsv():
    i = 0
    fileGenerator = Path("data/java/googlejam4_src").glob("**/*.method")
    label = []
    gcj_funcs = pd.DataFrame(columns=['id','code'])
    for f in tqdm(fileGenerator):
        with open(f) as method:
            gcj_funcs = gcj_funcs.append(pd.Series([i+CONS,method.read()], index=gcj_funcs.columns), ignore_index=True)
        label.append(int(str(f).split(os.sep)[-2]))
        i+=1
    labels = pd.Series(label).value_counts(normalize=True)
    labels.to_pickle("data/java/closs_test/labels_rate.pkl")
    gcj_funcs.to_csv("data/java/gcj_funcs_all.csv", index=False)
    pairs = pd.DataFrame(columns=['id1','id2','label'])
    for i in trange(len(label)):
        for j in range(i+1,len(label)):
            if label[i] == label[j]:
                pairs = pairs.append(pd.Series([i+CONS,j+CONS,label[i]], index=pairs.columns),ignore_index=True)
            else:
                pairs = pairs.append(pd.Series([i+CONS,j+CONS,0], index=pairs.columns),ignore_index=True)
    pairs.to_pickle("data/java/gcj_pair_ids.pkl")
    
if __name__ == '__main__':
    makeFuncTsv()