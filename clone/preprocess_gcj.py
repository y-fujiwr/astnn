from pathlib import Path
from tqdm import tqdm,trange
import os
import pandas as pd
import re

CONS = 30000000

stopword = ["scanner","print","reader","write","nextline","flush","close","file","locale"]
stopword_re = r'scanner|read|write|nextline|flush|close|file|locale|print'
goword_re = r'\Wif\W|\Wwhile\W|\Wfor\W|\Wmain\W|\Wcatch\W'

def makeFuncTsv():
    os.makedirs("data/gcj", exist_ok=True)
    os.makedirs("data/java/cross_test", exist_ok=True)
    i = 0
    fileGenerator = Path("data/googlejam4_src").glob("**/*.main")
    label = []
    gcj_funcs = pd.DataFrame(columns=['id','code'])
    for f in tqdm(fileGenerator):
        with open(f) as method:
            nest = 0
            delnest = []
            lines = method.readlines()
            output = ""
            for n in range(len(lines)):
                target = lines[n].lower()
                if len(re.findall("{\n",target)) > 0:
                    nest += 1
                if len(re.findall("}\n",target)) > 0:
                    nest -= 1
                    if nest+1 in delnest:
                        delnest.pop()
                        continue
                    output = output + lines[n]
                elif len(re.findall("try {",target)) > 0:
                    delnest.append(nest)
                elif len(re.findall(stopword_re,target)) == 0 or n == 0 or len(re.findall(goword_re,target)) > 0:
                    output = output + lines[n]
            gcj_funcs = gcj_funcs.append(pd.Series([i+CONS,output], index=gcj_funcs.columns), ignore_index=True)
        label.append(int(str(f).split(os.sep)[-2]))
        i+=1
    labels = pd.Series(label).value_counts(normalize=True)
    labels.to_pickle("data/java/cross_test/labels_rate.pkl")
    labels.to_pickle("data/gcj/labels_rate.pkl")
    gcj_funcs.to_csv("data/java/gcj_funcs_all_no_inout.csv", index=False)
    gcj_funcs.to_csv("data/gcj/gcj_funcs_all_no_inout.csv", index=False)
    pairs = pd.DataFrame(columns=['id1','id2','label'])
    for i in trange(len(label)):
        for j in range(i+1,len(label)):
            if label[i] == label[j]:
                pairs = pairs.append(pd.Series([i+CONS,j+CONS,label[i]], index=pairs.columns),ignore_index=True)
            elif (i+j)%20 == 0:
                pairs = pairs.append(pd.Series([i+CONS,j+CONS,0], index=pairs.columns),ignore_index=True)
    pairs.to_pickle("data/java/gcj_pair_ids.pkl")
    pairs.to_pickle("data/gcj/gcj_pair_ids.pkl")
    
if __name__ == '__main__':
    makeFuncTsv()