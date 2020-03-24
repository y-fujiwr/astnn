from pathlib import Path
from tqdm import tqdm,trange
import os
import pandas as pd
import re

CONS = 30000000

stopword = ["scanner","print","reader","write","nextline","flush","close","file","locale"]
stopword_re = r'scanner|read|write|nextline|flush|close|file|locale|print'
goword_re = r'\Wif\W|\Wwhile\W|\Wfor\W|\Wmain\W|\Wcatch\W'

def makeFuncTsv(target):
    os.makedirs("data/gcj", exist_ok=True)
    os.makedirs("data/java/cross_test", exist_ok=True)
    i = 0
    fileGenerator = Path(target).glob("**/*.main")
    label = []
    gcj_funcs = pd.DataFrame(columns=['id','code','path'])
    gcj_funcs_raw = pd.DataFrame(columns=['id','code'])
    for f in tqdm(fileGenerator):
        method = open(f)
        lines = method.readlines()

        if len(lines) > 0:
            nest = 0
            delnest = []
            output = ""
            output_raw = ""
            for n in range(len(lines)):
                output_raw = output_raw + lines[n]
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
            gcj_funcs = gcj_funcs.append(pd.Series([i+CONS,output,str(f)], index=gcj_funcs.columns), ignore_index=True)
            gcj_funcs_raw = gcj_funcs_raw.append(pd.Series([i+CONS,output_raw], index=gcj_funcs_raw.columns), ignore_index=True)
            label.append(int(str(f).split(os.sep)[-2]))
            i+=1
        method.close()
    labels = pd.Series(label).value_counts(normalize=True)
    labels.to_pickle("data/java/cross_test/labels_rate.pkl")
    labels.to_pickle("data/gcj/labels_rate.pkl")
    gcj_funcs.to_csv("data/gcj/gcj_funcs_all_no_inout.csv", index=False)
    gcj_funcs_raw.to_csv("data/gcj/gcj_funcs_all_raw.csv", index=False)
    
    pairs = pd.DataFrame(columns=['id1','id2','label'])
    for i in trange(len(label)):
        for j in range(i+1,len(label)):
            if label[i] == label[j]:
                pairs = pairs.append(pd.Series([i+CONS,j+CONS,label[i]], index=pairs.columns),ignore_index=True)
            else:
                pairs = pairs.append(pd.Series([i+CONS,j+CONS,0], index=pairs.columns),ignore_index=True)
    clone_pairs = pairs[pairs["label"]!=0]
    notclone_pairs = pairs[pairs["label"]==0].sample(n=len(clone_pairs), random_state=0)
    sampled_pairs = clone_pairs.append(notclone_pairs)
    sampled_pairs.to_pickle("data/gcj/gcj_pair_ids.pkl")
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--directory',type=str)
    args = parser.parse_args()
    makeFuncTsv(args.directory)