import pandas as pd
import torch
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from model import BatchProgramCC
from my_model import LSTM, BiLSTM, DNN, LSTM_ngram
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support,roc_curve,roc_auc_score
import os
import matplotlib.pyplot as plt
import re
from getCodeMetrics import getMetricsVec
import javalang
#from uploader import upload
warnings.filterwarnings('ignore')

source = None
dict_metrics = {}
def get_metrics(i):
    global dict_metrics
    if i in dict_metrics:
        return dict_metrics[i]
    else:
        s = source[source["id"]==i]["code"].iloc[0]
        v = getMetricsVec(s)
        dict_metrics[i] = v
        return v

def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    x1, x2, labels, id1, id2 = [], [], [], [], []
    for _, item in tmp.iterrows():
        if args.vector in ["trigram","monogram","bigram"]:
            code_x = re.sub("[\[\],\"\']","",str(item["code_x"])).split()
            code_y = re.sub("[\[\],\"\']","",str(item["code_y"])).split()

        elif args.model in ["lstm","bilstm"]:
            code_x = list(map(int,re.sub("[\[\],]","",str(item["code_x"])).split()))
            code_y = list(map(int,re.sub("[\[\],]","",str(item["code_y"])).split()))
            code_x = [min(i,MAX_TOKENS) for i in code_x]
            code_y = [min(i,MAX_TOKENS) for i in code_y]

        elif args.model in ["dnn"]:
            try:
                code_x = get_metrics(item["id1"])
                code_y = get_metrics(item["id2"])
            except javalang.parser.JavaSyntaxError:
                continue
        else:
            code_x = item["code_x"]
            code_y = item["code_y"]
        x1.append(code_x)
        x2.append(code_y)
        labels.append([float(item['label'])])
        id1.append(item['id1'])
        id2.append(item['id2'])
    return x1, x2, torch.FloatTensor(labels), id1, id2


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Choose a dataset:[c|java|gcj]")
    parser.add_argument('--lang')
    parser.add_argument('-g','--gpu', action='store_true')
    parser.add_argument('-r','--regression', action='store_true')
    parser.add_argument('-b','--batch_size', type=int, default=32)
    parser.add_argument('-e','--epoch', type=int, default=5)
    parser.add_argument('-m','--model',type=str,default="astnn")
    parser.add_argument('-v','--vector',type=str,default="w2v",help="choose: [w2v|lsi|trigram]")
    args = parser.parse_args()
    if not args.lang:
        print("No specified dataset")
        exit(1)
    root = 'data/'
    lang = args.lang
    if args.model in ["dnn"]:
        if lang in ["java"]:
            source = pd.read_csv(root+lang+'/bcb_funcs_all.tsv', sep='\t', header=None,names=["id","code"], encoding='utf-8')
        else:
            source = pd.read_csv(root+lang+'/{}_funcs_all.csv'.format(lang), encoding='utf-8')
    categories = 1
    if lang == 'java':
        categories = 12
    elif lang in 'gcj':
        categories = 12
    elif lang in 'check':
        categories = 2
    elif lang in 'sesame':
        categories = 2
    elif lang in 'csn':
        categories = 100
    print("Train for ", str.upper(lang))
    train_data = pd.read_pickle(root+lang+f'/train/blocks_{args.vector}.pkl').sample(frac=1)
    train_data = train_data[~(train_data['code_x'] == "[]") & ~(train_data['code_y'] == "[]")]
    test_data = pd.read_pickle(root+lang+f'/test/blocks_{args.vector}.pkl').sample(frac=1)
    test_data = test_data[~(test_data['code_x'] == "[]") & ~(test_data['code_y'] == "[]")]

    #word2vec
    if args.vector == "w2v":
        word2vec = Word2Vec.load(root+lang+"/train/embedding/node_w2v_128").wv
        MAX_TOKENS = word2vec.syn0.shape[0]
        EMBEDDING_DIM = word2vec.syn0.shape[1]
        embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
        embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    
    #lsi
    elif args.vector == "lsi":
        embeddings = np.load(root+lang+"/train/embedding/vec_lsi_256.npy").astype(np.float32)
        MAX_TOKENS = len(embeddings)
        EMBEDDING_DIM = 256
    #trigram
    elif args.vector == "trigram":
        EMBEDDING_DIM = 18929
    elif args.vector == "monogram":
        EMBEDDING_DIM = 26
    elif args.vector == "bigram":
        EMBEDDING_DIM = 728

    HIDDEN_DIM = 128
    ENCODE_DIM = 128
    LABELS = 1
    EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size
    if args.gpu == False:
        USE_GPU = False
    else:
        USE_GPU = True
    if args.vector in ["trigram","monogram","bigram"]:
        model = LSTM_ngram(EMBEDDING_DIM,HIDDEN_DIM,LABELS,BATCH_SIZE,args.vector,USE_GPU)
    elif args.model == "astnn":
        model = BatchProgramCC(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    elif args.model == "lstm":
        model = LSTM(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    elif args.model == "bilstm":
        model = BiLSTM(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    elif args.model == "dnn":
        model = DNN(LABELS,BATCH_SIZE,USE_GPU)
    else:
        print("No support")
        exit()
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    if args.regression:
        print("regression")
        loss_function = torch.nn.MSELoss()
        categories = 1
    else:
        loss_function = torch.nn.BCELoss()
        categories = 1

    precision, recall, f1 = 0, 0, 0
    print('Start training...')
    os.makedirs("data/{}/log".format(lang),exist_ok=True)
    for t in range(1, categories+1):
        result = open("data/{}/log/Type-{}.csv".format(lang,t),"w")
        result.write("id1,id2,trues,predicts,scores\n")
        if args.regression:
            train_data_t = train_data
            test_data_t = test_data
        elif lang in ['java','gcj','check','sesame','oreo','csn']:
            train_data_t = train_data#[train_data['label'].isin([t, 0])]
            train_data_t.loc[train_data_t['label'] > 0, 'label'] = 1

            test_data_t = test_data[test_data['label'].isin([t, 0])]
            test_data_t.loc[test_data_t['label'] > 0, 'label'] = 1
        else:
            train_data_t, test_data_t = train_data, test_data

        # training procedure
        for epoch in range(EPOCHS):
            print('Epoch {}'.format(epoch))
            start_time = time.time()
            # training epoch
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            i = 0
            while i < len(train_data_t):
                batch = get_batch(train_data_t, i, BATCH_SIZE)
                i += BATCH_SIZE
                train1_inputs, train2_inputs, train_labels, _, __ = batch
                if USE_GPU:
                    train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()

                model.zero_grad()
                model.batch_size = len(train_labels)
                if args.model == "astnn" and args.vector not in ["monogram","bigram","trigram"]:
                    model.hidden = model.init_hidden()
                output = model(train1_inputs, train2_inputs)
                loss = loss_function(output, Variable(train_labels))
                loss.backward()
                optimizer.step()
            
            print("Testing-%d..."%t)
            # testing procedure
            predicts = []
            trues = []
            outputs = []
            total_loss = 0.0
            total = 0.0
            i = 0
            while i < len(test_data_t):
                batch = get_batch(test_data_t, i, BATCH_SIZE)
                i += BATCH_SIZE
                test1_inputs, test2_inputs, test_labels, id1_batch, id2_batch = batch
                if USE_GPU:
                    test_labels = test_labels.cuda()
                model.batch_size = len(test_labels)
                #model.hidden = model.init_hidden()
                output = model(test1_inputs, test2_inputs)

                loss = loss_function(output, Variable(test_labels))

                # calc testing acc
                outputs.extend(output.data.cpu().numpy())
                predicted = (output.data > 0.5).cpu().numpy()
                predicts.extend(predicted)
                if args.regression:
                    trues.extend(np.where(test_labels.cpu().numpy() >= 1, 1, 0))
                else:
                    trues.extend(test_labels.cpu().numpy())
                total += len(test_labels)
                total_loss += loss.data[0] * len(test_labels)
                for j in range(len(predicted)):
                    result.write("{},{},{},{},{}\n".format(id1_batch[j],id2_batch[j],test_labels.cpu().numpy()[j],predicted[j],output.data.cpu().numpy()[j]))

            if args.regression:
                fpr,tpr,thresholds = roc_curve(trues,outputs)
                print("ROC_score: {}".format(roc_auc_score(trues,outputs)))
                plt.plot(fpr, tpr, marker='o')
                plt.xlabel('FPR: False positive rate')
                plt.ylabel('TPR: True positive rate')
                plt.grid()
                plt.savefig('data/{}/log/roc_curve_epoch{}.png'.format(lang,epoch))

            elif lang in ['java','gcj','check','sesame','oreo']:
                if lang in 'java':
                    weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
                elif lang in ['gcj']:
                    weights = pd.read_pickle("data/gcj/labels_rate.pkl")
                elif lang in 'check':
                    weights = [0, 0.962, 0.038]
                elif lang in 'sesame':
                    weights = [0, 0.75,0.25]
                elif lang in 'oreo':
                    weights = [0,1.0]
                p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')
                precision += weights[t] * p
                recall += weights[t] * r
                f1 += weights[t] * f
                print("Type-" + str(t) + ": " + str(p) + " " + str(r) + " " + str(f))
            else:
                precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
            #result = pd.DataFrame(trues,columns=['trues']).join(pd.DataFrame(predicts,columns=['predicts']))
            #result.to_csv("data/{}/log/Type-{}.csv".format(lang,t))
            

    #print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))
    os.makedirs("model", exist_ok=True)
    modelname = "model/{}_{}_{}.model".format(lang,args.regression,args.model)
    torch.save(model.state_dict(), modelname)
    #upload("model",modelname)