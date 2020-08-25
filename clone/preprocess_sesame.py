import pandas as pd
import random
df = pd.read_csv("data/sesame/sampled-pairs.csv")
df["label"] = 2
df.loc[df['sim']<0.5, 'label'] = 1
df2 = df.drop(["pairid","project1","file1","method1","project2","file2","method2","sim","sim_tok"],axis=1)
df3 = pd.read_csv("data/sesame/internal_filtered_methoddocs.csv")

df2.columns=["id1","id2","label"]
while len(df2) < len(df) * 2:
    a = random.randint(0,len(df3)-1)
    b = random.randint(0,len(df3)-1)
    if a==b:
        continue
    elif a > b:
        temp = a
        a = b
        b = temp
    df2 = df2.append(pd.Series([a,b,0],index=["id1","id2","label"]),ignore_index=True)
df2 = df2.drop_duplicates(subset=["id1","id2"])
df2.to_pickle("data/sesame/sesame_pair_ids.pkl")
exit()
df4 = pd.DataFrame(columns=['id','code'])
df3 = df3.drop(["project_id","kwset"],axis=1)
for _, t in df3.iterrows():
    try:
        s = open("data/sesame/src/{}.{}".format(t[1].split("/")[-1].split(".")[0], t[2].split("(")[0].split(".")[-1])).read()
        df4 = df4.append(pd.Series([t[0],s],index=df4.columns), ignore_index=True)
    except FileNotFoundError:
        print("{},{}".format(t[1],t[2]))
        pass
df4.to_csv("data/sesame/sesame_funcs_all.csv",index=False)