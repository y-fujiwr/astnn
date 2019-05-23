import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

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
        path = self.root+self.language+'/'+output_file
        if self.language in ['java','gcj'] and os.path.exists(path) and os.path.exists(self.root+self.language+"/ast_cross.pkl") and option == 'existing':
            source = pd.read_pickle(path)
            source_cross = pd.read_pickle(self.root+self.language+"/ast_cross.pkl")
        elif os.path.exists(path) and option == 'existing':
            source = pd.read_pickle(path)     
        else:
            if self.language is 'c':
                from pycparser import c_parser
                parser = c_parser.CParser()
                source = pd.read_pickle(self.root+self.language+'/programs.pkl')
                source.columns = ['id', 'code', 'label']
                source['code'] = source['code'].apply(parser.parse)
                source.to_pickle(path)
            elif self.language in ['java','gcj','check']:
                import javalang
                def parse_program(func):
                    tokens = javalang.tokenizer.tokenize(func)
                    parser = javalang.parser.Parser(tokens)
                    tree = parser.parse_member_declaration()
                    return tree
                if self.language in 'java':
                    source = pd.read_csv(self.root+self.language+'/bcb_funcs_all.tsv', sep='\t', header=None, encoding='utf-8')
                elif self.language in 'gcj':
                    source = pd.read_csv(self.root+self.language+'/gcj_funcs_all.csv', encoding='utf-8', engine='python')
                elif self.language in 'check':
                    source = pd.read_csv(self.root+self.language+'/check_funcs_all.csv', encoding='utf-8', engine='python')
                source.columns = ['id', 'code']
                source['code'] = source['code'].apply(parse_program)
                source.to_pickle(path)
                if self.language in 'java':
                    source_cross = pd.read_csv(self.root+self.language+'/gcj_funcs_all.csv', encoding='utf-8', engine='python')
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
        pairs = pd.read_pickle(self.root+self.language+'/'+filename)
        self.pairs = pairs
        if self.language is 'java':
            pairs_cross = pd.read_pickle(self.root+self.language+'/'+"gcj_pair_ids.pkl")
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
            input_file = self.train_file_path
        pairs = pd.read_pickle(input_file)
        train_ids = pairs['id1'].append(pairs['id2']).unique()

        trees = self.sources.set_index('id',drop=False).loc[train_ids]
        if not os.path.exists(data_path+'train/embedding'):
            os.mkdir(data_path+'train/embedding')
        if self.language is 'c':
            sys.path.append('../')
            from prepare_data import get_sequences as func
        elif self.language in ['java', 'gcj', 'check']:
            from utils import get_sequence as func

        def trans_to_sequences(ast):
            sequence = []
            func(ast, sequence)
            return sequence
        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        # trees.to_csv(data_path+'train/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
        w2v.save(data_path+'train/embedding/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self):
        if self.language is 'c':
            from prepare_data import get_blocks as func
        elif self.language in ['java', 'gcj', 'check']:
            from utils import get_blocks_v1 as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.root+self.language+'/train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
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

        df.to_pickle(self.root+self.language+'/'+part+'/blocks.pkl')
        df.to_csv(self.root+self.language+'/'+part+'/blocks.csv')

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl',option='existing')
        print('read id pairs...')
        if self.language is 'c':
            self.read_pairs('oj_clone_ids.pkl')
        elif self.language is 'java':
            self.read_pairs('bcb_pair_ids.pkl')
        elif self.language in 'gcj':
            self.read_pairs('gcj_pair_ids.pkl')
        elif self.language in 'check':
            self.read_pairs('check_pair_ids.pkl')
        print('split data...')
        self.split_data()
        print('train word embedding...')
        self.dictionary_and_embedding(None,128)
        print('generate block sequences...')
        self.generate_block_seqs()
        print('merge pairs and blocks...')
        self.merge(self.train_file_path, 'train')
        self.merge(self.dev_file_path, 'dev')
        self.merge(self.test_file_path, 'test')
        if self.language in ['java','gcj']:
            self.merge_cross(self.cross_test_file_path, 'cross_test')

import argparse
parser = argparse.ArgumentParser(description="Choose a dataset:[c|java|gcj]")
parser.add_argument('--lang')
args = parser.parse_args()
if not args.lang:
    print("No specified dataset")
    exit(1)
ppl = Pipeline('3:1:1', 'data/', str(args.lang))
ppl.run()


