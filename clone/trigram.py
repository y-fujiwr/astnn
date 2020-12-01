import numpy as np
import re
import pickle

dictionary = {}
signdict = {
    "+": "plus",
    "-": "minus",
    "*": "multiply",
    "/": "divide",
    "%": "remind",
    "++": "increment",
    "--": "decrement",
    "=": "substitute",
    "+=": "plussubstitute",
    "-=": "minussubstitute",
    "*=": "multiplysubstitute",
    "/=": "devidesubstitute",
    "%=": "remindsubstitute",
    "&": "and",
    "|": "or",
    "^": "xor",
    "!": "not",
    "==": "equal",
    "&&": "andand",
    "||": "oror",
    "!=": "notequal",
    "<": "lessthan",
    "<=": "lessequal",
    ">": "morethan",
    ">=": "moreequal",
    "<<": "shiftleft",
    ">>": "shiftright",
    ">>>": "shiftright",
    "&=": "andsubstitute",
    "^=": "xorsubstitute",
    "|=": "orsubstitute",
    "<<=": "shiftleftsubstitute",
    ">>=": "shiftrightsubstitute",
    ">>>=": "shiftrightsubstitute",
}
def get_signdict():
    return signdict

def save_dict():
    with open("chartrigram_dictionary.pkl","wb") as f:
        pickle.dump(dictionary,f)

def load_dict():
    global dictionary
    with open("chartrigram_dictionary.pkl","rb") as f:
        dictionary = pickle.load(f)

def trigram(string):
    l = np.zeros(18929)
    for i in range(len(string)-2):
        l[calculate_index(string[i:i+3])] += 1
    return l

def calculate_index(s):
    if s[0] == "^":
        if s[-1] == "$":
            return -1
        return alpha2num(s[1:])
    elif s[-1] == "$":
        return alpha2num(s[:-1]) + 18252
    else:
        return alpha2num(s) + 676

def alpha2num(alpha):
    num=0
    for index, item in enumerate(list(alpha)):
        num += pow(26,len(alpha)-index-1)*(ord(item)-ord('a'))
    return num

def getVector(string):
    string = "^" + re.sub(r"[^a-z]",r"",string.lower()) + "$"
    if string in dictionary.keys():
        return dictionary[string]
    else:
        v = trigram(string)
        dictionary[string] = v
        return v
