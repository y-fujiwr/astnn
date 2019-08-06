import pandas as pd
import torch
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from model import BatchProgramCC
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
import os
warnings.filterwarnings('ignore')



def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['code_x'])
        x2.append(item['code_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Choose a dataset:[c|java|gcj]")
    parser.add_argument('--lang')
    parser.add_argument('-g','--gpu', action='store_true')
    parser.add_argument('-b','--batch_size', type=int, default=32)
    parser.add_argument('-e','--epoch', type=int, default=5)
    args = parser.parse_args()
    if not args.lang:
        print("No specified dataset")
        exit(1)
    root = 'data/'
    lang = args.lang
    categories = 1
    if lang == 'java':
        categories = 5
    elif lang in 'gcj':
        categories = 12
    elif lang in 'check':
        categories = 2
    elif lang in 'sesame':
        categories = 2
    print("Train for ", str.upper(lang))
    train_data = pd.read_pickle(root+lang+'/train/blocks.pkl').sample(frac=1)
    train_data = train_data[~(train_data['code_x'] == "[]") & ~(train_data['code_y'] == "[]")]
    test_data = pd.read_pickle(root+lang+'/test/blocks.pkl').sample(frac=1)
    test_data = test_data[~(test_data['code_x'] == "[]") & ~(test_data['code_y'] == "[]")]

    word2vec = Word2Vec.load(root+lang+"/train/embedding/node_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 128
    ENCODE_DIM = 128
    LABELS = 1
    EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size
    if args.gpu == False:
        USE_GPU = False
    else:
        USE_GPU = True

    model = BatchProgramCC(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()

    precision, recall, f1 = 0, 0, 0
    print('Start training...')
    os.makedirs("data/{}/log".format(lang),exist_ok=True)
    for t in range(1, categories+1):
        if lang in ['java','gcj','check','sesame','oreo']:
            train_data_t = train_data[train_data['label'].isin([t, 0])]
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
                train1_inputs, train2_inputs, train_labels = batch
                if USE_GPU:
                    train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()

                model.zero_grad()
                model.batch_size = len(train_labels)
                model.hidden = model.init_hidden()
                output = model(train1_inputs, train2_inputs)

                loss = loss_function(output, Variable(train_labels))
                loss.backward()
                optimizer.step()
            print("Testing-%d..."%t)
            # testing procedure
            predicts = []
            trues = []
            total_loss = 0.0
            total = 0.0
            i = 0
            while i < len(test_data_t):
                batch = get_batch(test_data_t, i, BATCH_SIZE)
                i += BATCH_SIZE
                test1_inputs, test2_inputs, test_labels = batch
                if USE_GPU:
                    test_labels = test_labels.cuda()
                model.batch_size = len(test_labels)
                model.hidden = model.init_hidden()
                output = model(test1_inputs, test2_inputs)

                loss = loss_function(output, Variable(test_labels))

                # calc testing acc
                predicted = (output.data > 0.5).cpu().numpy()
                predicts.extend(predicted)
                trues.extend(test_labels.cpu().numpy())
                total += len(test_labels)
                total_loss += loss.data[0] * len(test_labels)
            if lang in ['java','gcj','check','sesame','oreo']:
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
            result = pd.DataFrame(trues,columns=['trues']).join(pd.DataFrame(predicts,columns=['predicts']))
            result.to_csv("data/{}/log/Type-{}.csv".format(lang,t))

    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/{}.model".format(lang))