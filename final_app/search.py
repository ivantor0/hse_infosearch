import re
import os
import time
import pickle
import logging
import numpy as np
import tensorflow as tf
import pymystem3

from collections.abc import Iterable
from scipy.sparse import csr_matrix
from random import choice
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine

from index import Indexer
from timedlogger import timelogged
from elmo_model import elmo_model
from elmo_helpers import tokenize, get_elmo_vectors, load_elmo_embeddings
from bert_serving.client import BertClient



class Loader:
    def __init__(self, index_path="./index"):
        self._models = ["tfidf", "bm25", "fasttext", "elmo", "bert"]
        self.index_path = index_path
        self.models = self.load_models()
        self._models_to_index = None
        if not self.check_index_dir():
            indexer = Indexer(index_path=self.index_path, models=self.models)
            if not self._models_to_index:
                self._models_to_index = self._models
            self._models += indexer.index(models=self._models_to_index)

    def check_index_dir(self):
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
        if not os.path.exists(os.path.join(self.index_path, "latest_index")):
            build_hash = self._make_hash()
            while build_hash in os.listdir(self.index_path):
                build_hash = self._make_hash()
            with open(os.path.join(self.index_path, "latest_index"), "w") as pf:
                pf.write(build_hash)
        else:
            with open(os.path.join(self.index_path, "latest_index"), "r") as pf:
                build_hash = pf.read()
        self.index_path = os.path.join(self.index_path, build_hash)
        if os.path.exists(self.index_path):
            models = [model for model in self._models
                      if model+".pickle" not in os.listdir(self.index_path)]
            if models:
                self._models_to_index = models
                return False
            return True
        else:
            os.mkdir(self.index_path)
            return False
 
    def _make_hash(self, nsymb=6):
        symbols = [chr(i) for i in range(ord("0"), ord("9")+1)] + \
                  [chr(i) for i in range(ord("a"), ord("z")+1)]
        result = "".join([choice(symbols) for _ in range(nsymb)])
        return result

    def load_index_structs(self):
        index_structs = {}
        for pfile in os.listdir(self.index_path):
            with open(os.path.join(self.index_path, pfile), "rb") as inp:
                index_structs[pfile[:-7]] = pickle.load(inp)
        return index_structs

    def load_models(self):
        models = {
            "fasttext": KeyedVectors.load("./models/fasttext/model.model"),
            "bert": BertClient(),
            "elmo": elmo_model()
        }
        return models

    def load_orig_strings(self):
        with open(os.path.join(self.index_path, "orig_strings.pickle"), "rb") as pfile:
            orig_strings = pickle.load(pfile)
        return orig_strings


class Searcher:

    def __init__(self, index_path="./index"):
        loader = Loader(index_path)
        self.orig_strings = loader.load_orig_strings()
        self._load_lemmatizer()
        self.index_structs = loader.load_index_structs()
        self.models = loader.load_models()
        # self.vocab = vocab
        self._schemae = {
            "Tf-idf": "tf-idf",
            "BM25": "bm25",
            "FastText": "fasttext",
            "ELMo": "elmo",
            "BERT": "bert"
        }

    def __getitem__(self, attr):
        self._schema = self._schemae[attr]
        return self

    def _load_lemmatizer(self):
        self._m = pymystem3.mystem.Mystem()

    def _preprocess_string(self, instr):
        if isinstance(instr, str):
            pass
        elif isinstance(instr, Iterable):
            instr = " ".join(instr)
        else:
            instr = str(instr)
        return [word for word in self._m.lemmatize(instr)
                if re.search(r'\w', word)]

    def search(self, query, top=10):
        _methods = {
            "tf-idf": self._search_tfidf,
            "bm25": self._search_bm25,
            "fasttext": self._search_fasttext,
            "elmo": self._search_elmo,
            "bert": self._search_bert
        }
        match = _methods[self._schema](query, top=10)
        match = ((m[0], "{0:.2f}".format(m[1])) for m in match)
        return match

    @timelogged("поиск TF-IDF")
    def _search_tfidf(self, query, top=10):
        logging.log(logging.INFO, "Запрос: "+query)
        query = self._preprocess_string(query)
        vocab, index = self.index_structs["tfidf"]
        query_row = [[0] for _ in range(index.shape[1])]
        for word in query:
            if word in vocab:
                query_row[vocab.index(word)] = [1]
        match = np.dot(index, csr_matrix(query_row)).toarray().flatten()
        match_dict = {self.orig_strings[i]: score for i, score in enumerate(match)}
        match = tuple(sorted(match_dict.items(),
                             key=lambda x: x[1], reverse=True)[:top])
        return match

    @timelogged("поиск BM25")
    def _search_bm25(self, query, top=10):
        logging.log(logging.INFO, "Запрос: "+query)
        query = self._preprocess_string(query)
        vocab, index = self.index_structs["bm25"]
        query_row = [[0] for _ in range(index.shape[1])]
        for word in query:
            if word in vocab:
                query_row[vocab[word]] = [1]
        match = np.dot(index, csr_matrix(query_row)).toarray().flatten()
        match_dict = {self.orig_strings[i]: score for i, score in enumerate(match)}
        match = tuple(sorted(match_dict.items(),
                             key=lambda x: x[1], reverse=True)[:top])
        return match

    @timelogged("поиск FastText")
    def _search_fasttext(self, query, top=10):
        logging.log(logging.INFO, "Запрос: "+query)
        query = self._preprocess_string(query)
        _, index = self.index_structs["fasttext"]
        vec_size = self.models["fasttext"].vector_size
        entity = np.zeros(vec_size)
        for word in query:
            try:
                entity += self.models["fasttext"][word]
            except AttributeError:
                pass
        if len(query):
            entity = entity / len(query)

        den = np.sqrt(np.einsum('ij,ij->i', index, index) * np.einsum('j,j', entity, entity))
        match = index.dot(entity) / den.flatten()

        match_dict = {self.orig_strings[i]: 1 - score for i, score in enumerate(match)}
        match = tuple(sorted(match_dict.items(), key=lambda x: x[1])[:top])
        return match

    @timelogged("поиск ELMo")
    def _search_elmo(self, query, top=10):
        logging.log(logging.INFO, "Запрос: "+query)
        query = self._preprocess_string(query)
        _, index = self.index_structs["elmo"]
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                              gpu_options=gpu_options)) as sess:
            with tf.device('/gpu:0'):
                # It is necessary to initialize variables once before running inference.
                sess.run(tf.global_variables_initializer())

                start = time.time()
                elmo_vectors = get_elmo_vectors(
                    sess, [query], self.models["elmo"].batcher, self.models["elmo"].sentence_character_ids,
                    self.models["elmo"].elmo_sentence_input)

                # Due to batch processing, the above code produces for each sentence
                # the same number of token vectors, equal to the length of the longest sentence
                # (the 2nd dimension of the elmo_vector tensor).
                # If a sentence is shorter, the vectors for non-existent words are filled with zeroes.
                # Let's make a version without these redundant vectors:

                cropped_vectors = []
                for vect, sent in zip(elmo_vectors, [query]):
                    cropped_vector = vect[:len(sent), :]
                    cropped_vectors.append(cropped_vector)

                vec = cropped_vectors[0]
                entity = np.sum(vec, axis=0) / len(vec)
                den = np.sqrt(np.einsum('ij,ij->i', index, index) * np.einsum('j,j', entity, entity))
                match = index.dot(entity) / den.flatten()

                match_dict = {self.orig_strings[i]: 1 - score for i, score in enumerate(match)}
                match = tuple(sorted(match_dict.items(), key=lambda x: x[1])[:top])
                print(query)
                return match

    @timelogged("поиск BERT")
    def _search_bert(self, query, top=10):
        logging.log(logging.INFO, "Запрос: "+query)
        if not query:
            entity = np.array([0] * 728)
        else:
            _, index = self.index_structs["bert"]
            entity = self.models["bert"].encode([query])[0]

        den = np.sqrt(np.einsum('ij,ij->i', index, index) * np.einsum('j,j', entity, entity))
        match = index.dot(entity) / den.flatten()

        match_dict = {self.orig_strings[i]: 1 - score for i, score in enumerate(match)}
        match = tuple(sorted(match_dict.items(), key=lambda x: x[1])[:top])
        return match



if __name__ == "__main__":
    searcher = Searcher()
    print(searcher["ELMo"].search("Какая завтра погода?"))
