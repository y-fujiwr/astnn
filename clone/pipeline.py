import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import csv
csv.field_size_limit(1000000000)
sys.setrecursionlimit(100000)
from time import time
import numpy as np
from tqdm import trange
import javalang
import lsi
from matching_w2v_vocab import importWordVocab, searchVocab
import pickle

cross_only = True
cross_project = "gcj"

class Pipeline:
    def __init__(self,  ratio, root, language):
        self.ratio = ratio
        self.root = root
        self.language = language
        self.sources = None
        self.source_cross = None
        self.blocks = None
        self.blocks_cross = None
        self.pairs = None
        self.pairs_cross = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.cross_test_file_path = None
        self.size = None

    # parse source code
    def parse_source(self, output_file, option):
        print(f"cross_only = {cross_only}")
        path = self.root+self.language+'/'+output_file
        if not cross_only and self.language in ['java','gcj'] and os.path.exists(path) and os.path.exists(self.root+self.language+"/ast_cross.pkl") and option == 'existing':
            source = pd.read_pickle(path)
            source_cross = pd.read_pickle(self.root+self.language+"/ast_cross.pkl")
        elif not cross_only and os.path.exists(path) and option == 'existing' and self.language not in ['check']:
            source = pd.read_pickle(path)     
        else:
            if self.language is 'c':
                from pycparser import c_parser
                parser = c_parser.CParser()
                source = pd.read_pickle(self.root+self.language+'/programs.pkl')
                source.columns = ['id', 'code', 'label']
                source['code'] = source['code'].apply(parser.parse)
                source.to_pickle(path)
            elif self.language in ['java','gcj','sort','check','sesame','oreo','csn','roy']:
                def parse_program(func):
                    try:
                        tokens = javalang.tokenizer.tokenize(func)
                        parser = javalang.parser.Parser(tokens)
                        tree = parser.parse_member_declaration()
                        return tree
                    except:
                        return "parseError"
                if cross_only:
                    source = pd.read_pickle(path)
                else:
                    if self.language in 'java':
                        source = pd.read_csv(self.root+self.language+'/bcb_funcs_all.tsv', sep='\t', header=None, encoding='utf-8')
                        source.columns = ['id', 'code']
                    else:
                        source = pd.read_csv(self.root+self.language+'/{}_funcs_all.csv'.format(self.language), encoding='utf-8')
                
                    if self.language not in ['oreo']:
                        source['code'] = source['code'].apply(parse_program)
                    source.to_pickle(path)
                if self.language in 'java':
                    source_cross = pd.read_csv(self.root+f'{cross_project}/{cross_project}_funcs_all.csv', encoding='utf-8', engine='python')
                    source_cross['code'] = source_cross['code'].apply(parse_program)
                    source_cross.to_pickle(self.root+self.language+"/ast_cross.pkl")
                elif self.language in 'gcj':
                    source_cross = pd.read_csv(self.root+'java/bcb_funcs_all.tsv', sep='\t', header=None, encoding='utf-8')
                    source_cross.columns = ['id','code']
                    source_cross['code'] = source_cross['code'].apply(parse_program)
                    source_cross.to_pickle(self.root+self.language+"/ast_cross.pkl")                        
        self.sources = source
        if self.language in ['java', 'gcj']:
            self.source_cross = source_cross
        return source

    # create clone pairs
    def read_pairs(self, filename):
        parse_error = list(self.sources[self.sources['code'].isin(['parseError'])].id)
        pairs = pd.read_pickle(self.root+self.language+'/'+filename)
        pairs = pairs[(~pairs['id1'].isin(parse_error)) & (~pairs['id2'].isin(parse_error))].dropna(how='any')
        self.pairs = pairs
        if self.language in 'java':
            pairs_cross = pd.read_pickle(self.root+f"{cross_project}/{cross_project}_pair_ids.pkl")
            self.pairs_cross = pairs_cross
        elif self.language in 'gcj':
            pairs_cross = pd.read_pickle(self.root+"java/bcb_pair_ids.pkl")
            self.pairs_cross = pairs_cross

    # split data for training, developing and testing
    def split_data(self):
        data_path = self.root+self.language+'/'
        data = self.pairs
        data_cross = self.pairs_cross
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)

        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split]
        dev = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        if not cross_only and not os.path.exists(f"{data_path}train/train_.pkl"):
            train_path = data_path+'train/'
            check_or_create(train_path)
            self.train_file_path = train_path+'train_.pkl'
            train.to_pickle(self.train_file_path)

            dev_path = data_path+'dev/'
            check_or_create(dev_path)
            self.dev_file_path = dev_path+'dev_.pkl'
            dev.to_pickle(self.dev_file_path)

            test_path = data_path+'test/'
            check_or_create(test_path)
            self.test_file_path = test_path+'test_.pkl'
            test.to_pickle(self.test_file_path)
        
        self.train_file_path = data_path + "train/train_.pkl"
        self.dev_file_path = data_path + "dev/dev_.pkl"
        self.test_file_path = data_path + "test/test_.pkl"

        if self.language in ['java','gcj']:
            cross_test_path = data_path + 'cross_test/'
            check_or_create(cross_test_path)
            self.cross_test_file_path = cross_test_path+'cross_test_.pkl'
            data_cross.to_pickle(self.cross_test_file_path)
            data_cross.to_csv(cross_test_path + "cross_test_.csv")

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        data_path = self.root+self.language+'/'
        if not input_file:
            input_file = "data/{}/train/train_.pkl".format(self.language)#self.train_file_path
        pairs = pd.read_pickle(input_file)
        train_ids = pairs['id1'].append(pairs['id2']).unique()
        trees = self.sources.set_index('id',drop=False).loc[train_ids]
        if self.language in ['oreo']:
            self.x = 0
            def parse_program(func):
                try:
                    self.x += 1
                    if self.x%1000 == 0:
                        print(self.x)
                    tokens = javalang.tokenizer.tokenize(func)
                    parser = javalang.parser.Parser(tokens)
                    tree = parser.parse_member_declaration()
                    return tree
                except:
                    return "parseError"
            trees['code'] = trees['code'].apply(parse_program)
        if not os.path.exists(data_path+'train/embedding'):
            os.mkdir(data_path+'train/embedding')
        if self.language is 'c':
            sys.path.append('../')
            from prepare_data import get_sequences as func
        elif self.language in ['java', 'gcj','sort', 'check','sesame','oreo','csn','roy']:
            from utils import get_sequence as func

        def trans_to_sequences(ast):
            sequence = []
            func(ast, sequence)
            return sequence
        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        str_corpus_lsi = [c for c in corpus]
        
        trees['code'] = pd.Series(str_corpus)#.lower())
        # trees.to_csv(data_path+'train/programs_ns.tsv')

        if not cross_only:
            from gensim.models.word2vec import Word2Vec
            w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
            w2v.save(data_path+'train/embedding/node_w2v_' + str(size))
        #lsi
        """
        lsi_model, lsi_dict = lsi.lsi(str_corpus_lsi,size)
        with open(f"{data_path}train/embedding/dictionary_lsi_{size}.pickle","wb") as fo:
            pickle.dump(lsi_dict, fo)
        embeddings = lsi_model.get_topics().T
        embeddings = np.append(embeddings, [[0.0]*size], axis=0)
        np.save(f'{data_path}train/embedding/vec_lsi_{size}', embeddings)
        """

    # generate block sequences with index representations
    def generate_block_seqs(self):
        if self.language is 'c':
            from prepare_data import get_blocks as func
        elif self.language in ['java', 'gcj', 'sort','check', 'sesame','oreo','csn','roy']:
            from utils import get_blocks_v1 as func
        from gensim.models.word2vec import Word2Vec

        importWordVocab(self.root+self.language+'/train/embedding/node_w2v_' + str(self.size))
        word2vec = Word2Vec.load(self.root+self.language+'/train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]
        #lsi
        """
        with open(f"{self.root}{self.language}/train/embedding/dictionary_lsi_{self.size}.pickle","rb") as fi:
            lsi_dict = pickle.load(fi)
        """

        def tree_to_index(node):
            token = node.token
            #word2vec
            result = [vocab[token].index if token in vocab else max_token]#searchVocab(token)]
            #lsi
            #result = [lsi_dict[token] if token in lsi_dict else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree
        if not cross_only:
            trees = pd.DataFrame(self.sources, copy=True)
            trees['code'] = trees['code'].apply(trans2seq)
            if 'label' in trees.columns:
                trees.drop('label', axis=1, inplace=True)
            self.blocks = trees
        if self.language in ['java','gcj']:
            trees_cross = pd.DataFrame(self.source_cross, copy=True)
            trees_cross['code'] = trees_cross['code'].apply(trans2seq)
            if 'label' in trees_cross.columns:
                trees_cross.drop('label', axis=1, inplace=True)
            self.blocks_cross = trees_cross


    # merge pairs
    def merge(self,data_path,part):
        pairs = pd.read_pickle(data_path)
        pairs['id1'] = pairs['id1'].astype(int)
        pairs['id2'] = pairs['id2'].astype(int)
        
        if self.language in ['oreo']:
            from utils import get_blocks_v1 as func
            from gensim.models.word2vec import Word2Vec
            word2vec = Word2Vec.load(self.root+self.language+'/train/embedding/node_w2v_' + str(self.size)).wv
            vocab = word2vec.vocab
            def parse_program(func):
                try:
                    self.x += 1
                    if self.x%1000 == 0:
                        print(self.x)
                    tokens = javalang.tokenizer.tokenize(func)
                    parser = javalang.parser.Parser(tokens)
                    tree = parser.parse_member_declaration()
                    return tree
                except:
                    return "parseError"
            def tree_to_index(node):
                token = node.token
                result = [vocab[token].index if token in vocab else searchVocab(token)]
                children = node.children
                for child in children:
                    result.append(tree_to_index(child))
                return result
            def trans2seq(r):
                blocks = []
                func(r, blocks)
                tree = []
                for b in blocks:
                    btree = tree_to_index(b)
                    tree.append(btree)
                return tree
            train_ids = pairs['id1'].append(pairs['id2']).unique()
            trees = self.sources.set_index('id',drop=False).loc[train_ids]
            trees = trees.rename(columns={'id':'id3'})
            self.x = 0
            trees['code'] = trees['code'].apply(parse_program)
            parse_error = list(trees[trees['code'].isin(['parseError'])].id3)
            pairs = pairs[(~pairs['id1'].isin(parse_error)) & (~pairs['id2'].isin(parse_error))].dropna(how='any')
            trees = trees[~trees['code'].isin(['parseError'])]
            trees['code'] = trees['code'].apply(trans2seq)
            df = pd.merge(pairs, trees, how='left', left_on='id1', right_on='id3')
            df = pd.merge(df, trees, how='left', left_on='id2', right_on='id3')
            df.drop(['id3_x', 'id3_y'], axis=1,inplace=True)
            df.dropna(inplace=True)
        else:
            df = pd.merge(pairs, self.blocks, how='left', left_on='id1', right_on='id')
            df = pd.merge(df, self.blocks, how='left', left_on='id2', right_on='id')
            df.drop(['id_x', 'id_y'], axis=1,inplace=True)
            df.dropna(inplace=True)        
        df.to_pickle(self.root+self.language+'/'+part+'/blocks.pkl')
        df.to_csv(self.root+self.language+'/'+part+'/blocks.csv')

    def merge_cross(self,data_path,part):
        pairs = pd.read_pickle(data_path)
        pairs['id1'] = pairs['id1'].astype(int)
        pairs['id2'] = pairs['id2'].astype(int)
        df = pd.merge(pairs, self.blocks_cross, how='left', left_on='id1', right_on='id')
        df = pd.merge(df, self.blocks_cross, how='left', left_on='id2', right_on='id')
        df.drop(['id_x', 'id_y'], axis=1,inplace=True)
        df.dropna(inplace=True)

        df.to_pickle(self.root+self.language+'/'+part+f'/{cross_project}_lsi.pkl')
        df.to_csv(self.root+self.language+'/'+part+f'/blocks.csv')

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl',option='existing')
        print('read id pairs...')
        if self.language in 'c':
            self.read_pairs('oj_clone_ids.pkl')
        elif self.language in 'java':
            self.read_pairs('bcb_pair_simast.pkl')
        elif self.language in 'oreo':
            self.read_pairs('oreo_pair_sim.pkl')
        elif self.language in 'sesame':
            self.read_pairs('sesame_pair_simast.pkl')
        else:
            self.read_pairs('{}_pair_ids.pkl'.format(self.language))
        print('split data...')
        self.split_data()
        print('train word embedding...')
        self.dictionary_and_embedding(None,128)
        if self.language in ['oreo']:
            self.blocks = self.sources
        else:
            print('generate block sequences...')
            self.generate_block_seqs()
        print('merge pairs and blocks...')
        if not cross_only:
            self.merge(self.train_file_path, 'train')
            self.merge(self.dev_file_path, 'dev')
            self.merge(self.test_file_path, 'test')
        if self.language in ['java','gcj']:
            self.merge_cross(self.cross_test_file_path, 'cross_test')
        if self.language in 'sort':
            x = pd.read_pickle("data/sort/train/blocks.pkl").append(pd.read_pickle("data/sort/dev/blocks.pkl")).append(pd.read_pickle("data/sort/test/blocks.pkl"))
            x.to_csv("data/sort/test/blocks_for_cross.csv")
            x.to_pickle("data/sort/test/blocks_for_cross.pkl")

import argparse
parser = argparse.ArgumentParser(description="Choose a dataset:[c|java|gcj]")
parser.add_argument('--lang')
parser.add_argument('--single','-s', action='store_false')
args = parser.parse_args()
if not args.lang:
    print("No specified dataset")
    exit(1)
cross_only = args.single
if args.lang in 'sort':
    ppl = Pipeline('1:1:1', 'data/', str(args.lang))
elif args.lang in 'java':
    ppl = Pipeline('998:1:1', 'data/', str(args.lang))
else:
    ppl = Pipeline('8:1:1', 'data/', str(args.lang))
ppl.run()


