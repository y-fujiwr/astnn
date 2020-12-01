from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.jaccard import JaccardMeasure
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher
from gensim.models.word2vec import Word2Vec

db = None
word2vec = None
max_token = None
searcher = None
def importWordVocab(path):
    global db
    global word2vec
    global max_token
    global searcher
    db = DictDatabase(CharacterNgramFeatureExtractor(2))
    model = Word2Vec.load(path).wv
    word2vec = model.vocab
    max_token = model.syn0.shape[0]
    for key in word2vec.keys():
        db.add(key)
    searcher = Searcher(db, JaccardMeasure())

def searchVocab(vocab):
    global db
    global word2vec
    global max_token
    global searcher
    results = searcher.ranked_search(vocab, 0.3)
    if len(results) > 0:
        return word2vec[results[0][1]].index
    else:
        return max_token

if __name__ == '__main__':
    importWordVocab("data/java/trains/embedding/node_w2v_128")
    print(type(searchVocab("sorting")))
