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

def monogram(ch):
    l = np.zeros(26)
    for i in range(len(ch)):
        l[ord(ch[i])-ord("a")] += 1
    return l

def getVector(string):
    string = re.sub(r"[^a-z]",r"",string.lower())
    if string in dictionary.keys():
        return dictionary[string]
    else:
        v = monogram(string)
        dictionary[string] = v
        return v
