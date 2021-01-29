import tkinter
from tkinter import ttk, messagebox
import pandas as pd
import os

target = "csn"

pairs = pd.read_csv(f"data/{target}/{target}_must_check.csv")
funcs = pd.read_csv(f"data/{target}/{target}_funcs_all.csv")
answer_frame = pd.DataFrame(columns=["id1","id2","label"])
inadequate_data = pd.DataFrame()
if os.path.isfile(f"data/{target}/answers_{target}.csv"):
    answer_frame = pd.read_csv(f"data/{target}/answers_{target}.csv")
    answerd = answer_frame.drop("label",axis=1)
    pairs = pairs.append(answerd).drop_duplicates(keep=False)
    #inadequate_data = pd.read_csv(f"data/{target}/inadequate_data_{target}.csv")

pointer = 0
#print(pairs.iloc[0])

main_win = tkinter.Tk()
main_win.title(f"Clone Judgement {pointer}/{len(pairs)}")
main_win.geometry("1500x1000")

main_frm = ttk.Frame(main_win)
main_frm.grid(column=0, row=0, sticky=tkinter.NSEW, padx=5, pady=10)

code1 = tkinter.Text(main_frm, width=110)
code2 = tkinter.Text(main_frm, width=110)

def app_clone():
    global answer_frame
    answer_frame = answer_frame.append(pd.Series([pairs.iloc[pointer]["id1"],pairs.iloc[pointer]["id2"],1], index=answer_frame.columns), ignore_index=True)
    next_pair()


def app_notclone():
    global answer_frame
    answer_frame = answer_frame.append(pd.Series([pairs.iloc[pointer]["id1"],pairs.iloc[pointer]["id2"],0], index=answer_frame.columns), ignore_index=True)
    next_pair()

def app_inadequate():
    global inadequate_data
    inadequate_data = inadequate_data.append(pairs.iloc[pointer])
    next_pair()

def app_exit():
    global inadequate_data
    #pairs.drop(range(pointer)).reset_index(drop=True).to_pickle(f"data/{target}/shuf_pair_{target}.pkl")
    #inadequate_data.to_csv(f"data/{target}/inadequate_data_{target}.csv", index=False)
    answer_frame.to_csv(f"data/{target}/answers_{target}.csv", index=False)
    messagebox.showinfo("save")

def next_pair():
    global pointer
    global code1
    global code2
    global main_win
    pointer += 1
    code1.delete("1.0","end")
    code2.delete("1.0","end")
    code1.insert("1.0",funcs[funcs["id"]==pairs.iloc[pointer]["id1"]]["code"].iloc[0])
    code2.insert("1.0",funcs[funcs["id"]==pairs.iloc[pointer]["id2"]]["code"].iloc[0])
    #print(pairs.iloc[pointer])
    main_win.title(f"Clone Judgement {pointer}/{len(pairs)}")

p_btn = ttk.Button(main_frm, text="Clone", command=app_clone)
n_btn = ttk.Button(main_frm, text="Not-Clone", command=app_notclone)
u_btn = ttk.Button(main_frm, text="Inadequate", command=app_inadequate)
e_btn = ttk.Button(main_frm, text="Save", command=app_exit)

code1.insert("1.0",funcs[funcs["id"]==pairs.iloc[0]["id1"]]["code"].iloc[0])
code2.insert("1.0",funcs[funcs["id"]==pairs.iloc[0]["id2"]]["code"].iloc[0])

code1.grid(column=0,row=0)
code2.grid(column=1,row=0)

p_btn.grid(column=0, row=1, sticky=tkinter.W)
n_btn.grid(column=0, row=1)
u_btn.grid(column=0, row=1, sticky=tkinter.E)
e_btn.grid(column=1, row=1)

main_win.mainloop()