import pandas as pd
import torch
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from model import BatchProgramCC
from my_model import LSTM,BiLSTM,DNN
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support,roc_curve,roc_auc_score
import os
import matplotlib.pyplot as plt
import re
from getCodeMetrics import getMetricsVec
import javalang
from uploader import upload
warnings.filterwarnings('ignore')

source = None
dict_metrics = {}
testlang = None
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
        if args.model in ["lstm","bilstm"]:
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
        if testlang == "bcb":
            try:
                item["label"] = 1 if float(item["label"]) > 0.0 else 0
            except ValueError:
                item["label"] = 1
        x1.append(code_x)
        x2.append(code_y)
        labels.append([item['label']])
        id1.append(item['id1'])
        id2.append(item['id2'])
    return x1, x2, torch.FloatTensor(labels), id1, id2


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Choose a dataset:[c|java|gcj]")
    parser.add_argument('--lang')
    parser.add_argument('-r','--regression', action='store_true')
    parser.add_argument('-t','--testfile', type=str , default="blocks.pkl")
    parser.add_argument('-m','--model',type=str,default="astnn")
    args = parser.parse_args()
    if not args.lang:
        print("No specified dataset")
        exit(1)
    root = 'data/'
    lang = args.lang
    for l in ["bcb","gcj","roy","sesame","csn"]:
        if l in args.testfile:
            testlang = l
            break
    print(testlang)
    if args.model in ["dnn"]:
        if testlang in ["bcb"]:
            source = pd.read_csv(root+lang+'/bcb_funcs_all.tsv', sep='\t', header=None, names=["id","code"], encoding='utf-8')
        else:
            source = pd.read_csv(root+testlang+'/{}_funcs_all.csv'.format(testlang), encoding='utf-8')
    categories = 1
    if lang == 'java':
        categories = 12
        if "csn" in args.testfile:
            categories = 100
            print(100)
        if "bcb" in args.testfile:
            categories = 1
    elif lang in ['gcj','oreo']:
        categories = 12
    elif lang in "sesame":
        categories = 2
    print("Train for ", str.upper(lang))
    test_data = pd.read_pickle(root+lang+'/cross_test/{}'.format(args.testfile)).sample(frac=1)
    #word2vec
    
    word2vec = Word2Vec.load(root+lang+"/train/embedding/node_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    """
    #lsi
    embeddings = np.load(root+lang+"/train/embedding/vec_lsi_128.npy").astype(np.float32)
    MAX_TOKENS = len(embeddings)
    EMBEDDING_DIM = 128
    """

    HIDDEN_DIM = 128
    ENCODE_DIM = 128
    LABELS = 1
    EPOCHS = 10
    BATCH_SIZE = 32
    USE_GPU = False
    if args.model == "astnn":
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
    model_name = "{}_{}_{}.model".format(lang,args.regression,args.model)
    model.load_state_dict(torch.load(f"model/{model_name}"))
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()

    precision, recall, f1 = 0, 0, 0
    print('Start testing...')
    resultdir = "data/{}/log_cross/{}_{}/".format(lang,args.testfile.split(".")[0],model_name)
    os.makedirs(resultdir,exist_ok=True)
    for t in range(categories+1):
        result = open("{}Type-{}.csv".format(resultdir,t),"w")
        result.write("id1,id2,trues,predicts,scores\n")
        if lang in ['java','gcj','oreo','sesame']:
            if testlang == "bcb":
                test_data_t = test_data
            else:
                test_data_t = test_data[test_data['label'].isin([t])]
                test_data_t.loc[test_data_t['label'] > 0, 'label'] = 1
            
        else:
            train_data_t, test_data_t = train_data, test_data
        print("Testing-%d..."%t)
        start = time.time()
        # testing procedure
        predicts = []
        trues = []
        scores = []
        id1 = []
        id2 = []
        i = 0
        while i < len(test_data_t):
            batch = get_batch(test_data_t, i, BATCH_SIZE)
            i += BATCH_SIZE
            test1_inputs, test2_inputs, test_labels, id1_batch, id2_batch = batch
            if USE_GPU:
                test_labels = test_labels.cuda()

            model.batch_size = len(test_labels)
            if args.model == "astnn":
                model.hidden = model.init_hidden()
            try:
                output = model(test1_inputs, test2_inputs)
                # calc testing acc
                predicted = (output.data > 0.5).cpu().numpy()
                scores.extend(output.data.numpy())
                """
                id1.extend(id1_batch)
                id2.extend(id2_batch)
                """
                trues.extend(test_labels.cpu().numpy())
                predicts.extend(predicted)
                for j in range(len(predicted)):
                    result.write("{},{},{},{},{}\n".format(id1_batch[j],id2_batch[j],test_labels.cpu().numpy()[j],predicted[j],output.data.numpy()[j]))
            except ValueError as e:
                print(e)
                print(f"{id1_batch},{id2_batch}")

        """
        if args.regression:
            fpr,tpr,thresholds = roc_curve(trues,scores)
            print("ROC_score: {}".format(roc_auc_score(trues,scores)))
            plt.plot(fpr, tpr, marker='o')
            plt.xlabel('FPR: False positive rate')
            plt.ylabel('TPR: True positive rate')
            plt.grid()
            plt.savefig('{}roc_curve_type{}.png'.format(resultdir,categories))
        """
        """
        if lang in ['java','gcj','oreo']:
            if lang in 'java':
                weights = [1] * 100#pd.read_pickle("data/java/cross_test/labels_rate.pkl")
            elif lang in ['gcj','oreo']:
                weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')
            precision += weights[t] * p
            recall += weights[t] * r
            f1 += weights[t] * f
            print("Type-" + str(t) + ": " + str(p) + " " + str(r) + " " + str(f))
            print("elapsed_time:{}[sec]".format(time.time()-start))
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
        """
        result.close()
        """
        result = pd.DataFrame(id1,columns=['id1']).join(pd.DataFrame(id2,columns=['id2'])).join(pd.DataFrame(trues,columns=['trues'])).join(pd.DataFrame(predicts,columns=['predicts'])).join(pd.DataFrame(scores,columns=['scores']))
        result.to_csv("data/{}/log_cross/Type-{}.csv".format(lang,t))
        """

    #print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))
    upload("result",resultdir[:-1])