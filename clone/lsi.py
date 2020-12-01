import glob
import os
import gensim
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from collections import defaultdict
import pandas as pd
import numpy as np
import pickle
import argparse

dictionary_lsi = None
vectors_lsi = None

def inverse_dict(d):
    return {v:k for k,v in d.items()}
def lsi(texts, dim):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    # lsi model
    lsi_model = models.LsiModel(corpus=corpus, id2word=dictionary, num_topics=dim)
    return lsi_model, inverse_dict(dictionary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--target',type=str)
    parser.add_argument('-l','--language',type=str,default='java')
    parser.add_argument('-d','--dimension',type=int,default='200')
    args = parser.parse_args()

    functions = pd.read_csv(f'{args.target}/functions.csv', names=('id', 'document'))

    texts = [[word for word in document.split() ] for document in functions['document']]

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text] for text in texts]
    
    lsi_model, dictionary = lsi(texts,args.dimension)

    with open(f"{args.target}/dictionary_lsi_{args.dimension}.pickle","wb") as fo:
        pickle.dump(inverse_dict(dictionary), fo)

    np.save(f'{args.target}/vec_lsi_{args.dimension}', lsi_model.get_topics().T)
    print(len(lsi_model.get_topics().T))