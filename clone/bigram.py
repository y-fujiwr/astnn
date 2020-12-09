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

def bigram(string):
    l = np.zeros(728)
    for i in range(len(string)-1):
        l[calculate_index(string[i:i+2])] += 1
    return l

def calculate_index(s):
    if s[0] == "^":
        return alpha2num(s[1:])
    elif s[-1] == "$":
        return alpha2num(s[:-1]) + 702
    else:
        return alpha2num(s) + 26

def alpha2num(alpha):
    num=0
    for index, item in enumerate(list(alpha)):
        print(index,item)
        num += pow(26,len(alpha)-index-1)*(ord(item)-ord('a'))
        print(num)
    return num

def getVector(string):
    string = "^" + re.sub(r"[^a-z]",r"",string.lower()) + "$"
    if string in dictionary.keys():
        return dictionary[string]
    else:
        v = bigram(string)
        dictionary[string] = v
        return v
