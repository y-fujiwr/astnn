import javalang
import javalang.ast as ast
from collections import defaultdict
import argparse
from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d','--directory',type=str)
parser.add_argument('-l','--language',type=str,default='java')
args = parser.parse_args()

funcs = pd.DataFrame(columns=['id','code'])
pairs = pd.DataFrame(columns=['id1','id2','label'])
funcid = 0

fileGenerator = Path(args.directory).glob("**/*.{}".format(args.language))

for f in tqdm(fileGenerator):
    text = "class a {\n"
    text = text + open(str(f)).read()
    text = text + "}\n"
    try:
        tokens = list(javalang.tokenizer.tokenize(text))
    except javalang.tokenizer.LexerError:
        continue

    positions = []

    #各メソッドの位置をpositionsに保存
    try:
        tree = javalang.parse.parse(text)
    except javalang.parser.JavaSyntaxError:
        continue
    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        positions.append(node.position)

    #javalangでうまくパースできないメソッドが含まれているクローンペアは除外
    if len(positions) != 2:
        continue

    methods = [[]]
    i = 0
    nest = 0
    for token in tokens:
        if i >= len(positions):
            break 
        if token.position >= positions[i]:
            if token.value == "{":
                nest += 1
            if nest >= 0:
                #書き込み
                methods[-1].append(token.value)
            if token.value == "}":
                nest -= 1
                if nest == 0:
                    i += 1
                    methods.append([])

    methodStr = []
    for m in methods:
        methodStr.append(" ".join(m))

    for m in methodStr[:-1]:
        funcs = funcs.append(pd.Series([funcid,m],index=funcs.columns), ignore_index=True)
        funcid += 1

funcs.to_csv("data/roy/roy_funcs_all.csv",index=False)
for j in range(0,len(funcs),2):
    pairs = pairs.append(pd.Series([j,j+1,1],index=pairs.columns),ignore_index=True)

num_pairs = len(pairs)
num_funcs = len(funcs)

#下の手順で作成した非クローンペアには，クローンペアが混ざる可能性がある
import random
while len(pairs) < num_pairs * 2:
    a = random.randint(0,num_funcs-1)
    b = random.randint(0,num_funcs-1)
    if a == b:
        continue
    elif a > b:
        temp = a
        a = b
        b = temp
    if b-a == 1 and a%2 == 0:
        continue
    pairs = pairs.append(pd.Series([a,b,0],index=pairs.columns),ignore_index=True)

pairs.to_pickle("data/roy/roy_pair_ids.pkl")