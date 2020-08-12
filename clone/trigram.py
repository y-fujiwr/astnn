import numpy as np

def trigram(string):
    string = "^" + string.lower() + "$"
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

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
