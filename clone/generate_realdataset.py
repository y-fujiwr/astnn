#ジャーナルの条件2に対応するための学習データ生成スクリプト
import pandas as pd
import sys
from tqdm import tqdm,trange
from notify import notify
import atexit
text = " ".join(sys.argv)
atexit.register(notify, f'【悲報】産総研の業務で動かしているスクリプト: {text} がエラーで終了しました。')

target_dataset = sys.argv[1:]

for dataset in target_dataset:
    if dataset == "java":
        pass
    else:
        methodtable = pd.read_csv(f"data/{dataset}/{dataset}_funcs_all.csv")
    if dataset == "gcj":
        def extractQuestionId(s):
            questionID = int(s.split("/")[-2])
            return questionID
        methodtable["qid"] = methodtable["path"].apply(extractQuestionId)
        methodtable["id"] = methodtable["id"]
        # 問題番号5以下だけ対象にする場合は以下を実行
        # methodtable = methodtable[methodtable["qid"] <= 5]
        # methodtable = methodtable[methodtable["qid"] > 3]
        real_pairs = pd.DataFrame(columns=['id1','id2','label'])
        for i in trange(len(methodtable)):
            for j in range(i+1,len(methodtable)):
                id1 = methodtable.iloc[i]["id"]
                id2 = methodtable.iloc[j]["id"]
                if methodtable.iloc[i]["qid"] == methodtable.iloc[j]["qid"]:
                    real_pairs = real_pairs.append(pd.Series([id1,id2,1], index=real_pairs.columns),ignore_index=True)
                else:
                    real_pairs = real_pairs.append(pd.Series([id1,id2,0], index=real_pairs.columns),ignore_index=True)

    elif dataset == "csn":
        pairs = pd.read_pickle(f"data/{dataset}/{dataset}_pair_ids.pkl")
        clones = pairs[pairs["label"] > 0]
        code_list = clones["id1"].append(clones["id2"]).drop_duplicates().reset_index(drop=True)
        real_pairs = pd.DataFrame(columns=['id1','id2','label'])
        must_check = pd.DataFrame(columns=['id1','id2'])
        for i in trange(len(code_list)):
            for j in range(i+1,len(code_list)):
                id1 = code_list[i]
                id2 = code_list[j]
                q1 = methodtable[methodtable["id"]==id1].iloc[0]["query"]
                q2 = methodtable[methodtable["id"]==id2].iloc[0]["query"]
                if q1 == q2:
                    if len(clones[(clones["id1"] == id1) & (clones["id2"] == id2)]) > 0 or len(clones[(clones["id1"] == id2) & (clones["id2"] == id1)]) > 0:
                        real_pairs = real_pairs.append(pd.Series([id1,id2,1], index=real_pairs.columns),ignore_index=True)
                    else:
                        must_check = must_check.append(pd.Series([id1,id2], index=must_check.columns),ignore_index=True)
                else:
                    real_pairs = real_pairs.append(pd.Series([id1,id2,0], index=real_pairs.columns),ignore_index=True)
        must_check.to_csv(f"data/{dataset}/{dataset}_must_check.csv",index=False)
    elif dataset == "sesame":
        pairs = pd.read_pickle(f"data/{dataset}/{dataset}_pair_ids.pkl")
        clones = pairs[pairs["label"] > 0]
        code_list = clones["id1"].append(clones["id2"]).drop_duplicates().reset_index(drop=True)
        real_pairs = pd.DataFrame(columns=['id1','id2','label'])
        must_check = pd.DataFrame(columns=['id1','id2'])
        for i in trange(len(code_list)):
            for j in range(i+1,len(code_list)):
                id1 = code_list[i]
                id2 = code_list[j]
                if len(clones[(clones["id1"] == id1) & (clones["id2"] == id2)]) > 0 or len(clones[(clones["id1"] == id2) & (clones["id2"] == id1)]) > 0:
                    real_pairs = real_pairs.append(pd.Series([id1,id2,1], index=real_pairs.columns),ignore_index=True)
                else:
                    real_pairs = real_pairs.append(pd.Series([id1,id2,0], index=real_pairs.columns),ignore_index=True)
    else:
        real_pairs = pd.DataFrame(columns=['id1','id2','label'])
        pairs = pd.read_pickle(f"data/{dataset}/{dataset}_pair_ids.pkl")
        clones = pairs[pairs["label"] > 0]
        for i in trange(len(methodtable)):
            for j in range(i+1,len(methodtable)):
                id1 = methodtable.iloc[i]["id"]
                id2 = methodtable.iloc[j]["id"]
                if len(clones[(clones["id1"] == id1) & (clones["id2"] == id2)]) > 0 or len(clones[(clones["id1"] == id2) & (clones["id2"] == id1)]) > 0:
                    real_pairs = real_pairs.append(pd.Series([id1,id2,1], index=real_pairs.columns),ignore_index=True)
                else:
                    real_pairs = real_pairs.append(pd.Series([id1,id2,0], index=real_pairs.columns),ignore_index=True)

    real_pairs.to_pickle(f"data/{dataset}/{dataset}_pair_ids_real.pkl")
    real_pairs.to_csv(f"data/{dataset}/{dataset}_pair_ids_real.csv",index=False)

atexit.unregister(notify)
atexit.register(notify,f"産総研で実行しているスクリプト: {text} が正常に終了しました。")
