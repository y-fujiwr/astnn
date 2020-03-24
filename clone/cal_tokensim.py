import pandas as pd
import os,sys,javalang
import numpy as np
from utils import get_sequence
import editdistance

def trans_to_sequences(ast):
    sequence = []
    get_sequence(ast, sequence)
    return sequence
def parse_program(func):
    try:
        tokens = javalang.tokenizer.tokenize(func)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        return tree
    except:
        return "parseError"
def main(lang):
    print("read funcs")
    source = pd.read_csv("data/{}/{}_funcs_all.csv".format(lang,lang),encoding="utf-8")

    print("read pairs")
    pairs = pd.read_pickle("data/{}/{}_pair_ids.pkl".format(lang,lang))

    print("extract positive data")
    pairs_pos = pairs[pairs["label"]>=1]

    print("remove unused sources")
    used_source_id = pairs['id1'].append(pairs['id2']).unique()
    source = source[source['id'].isin(used_source_id)]
    source = source.set_index("id")
    pairs["label"] = 0.0

    print("generate ast sequences")
    source['code'] = source['code'].apply(parse_program)
    source['code'] = source['code'].apply(trans_to_sequences)
    a = len(pairs)

    print("calculate editdistance")
    for i,j,index in zip(pairs_pos["id1"],pairs_pos["id2"],pairs_pos.index):
        nl = editdistance.eval(source['code'][i],source['code'][j])/(max(len(source['code'][i]),len(source['code'][j])) * 1.00)
        pairs.at[index,"label"] = 1-nl
        if index % 10000 == 0:
            print("{}/{}".format(index,a))
    pairs.to_pickle("data/{}/{}_pair_sim.pkl".format(lang,lang))
    
main("oreo")