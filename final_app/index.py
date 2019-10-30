import re
import os
import sys
import pickle
import csv
import time
import logging
import numpy as np
import tensorflow as tf
import pymystem3

from math import log
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
from timedlogger import timelogged
from elmo_model import elmo_model
from elmo_helpers import tokenize, get_elmo_vectors, load_elmo_embeddings
from bert_serving.client import BertClient


class DocumentReader:
    def get_docs(self, filepath="./data/question_pairs.csv", stop=100000):
        self.filepath = filepath
        if filepath == "./data/question_pairs.csv":
            return self._get_default_doc(stop)
        return self.get_from_txt(filepath, stop)

    def _get_default_doc(self, stop=100000):
        docs = []
        with open("./data/question_pairs.csv") as csvfile:
            docreader = csv.reader(csvfile)
            for i, row in enumerate(docreader):
                if i:
                    docs.append(row[2])
                if i > stop:
                    break
        return docs

    def get_from_txt(self, filepath, stop=100000):
        with open(filepath, "r", encoding="utf-8") as infile:
            docs = infile.read().split("\n")
        return docs[:stop]


class Indexer:

    def __init__(self, filepath="./data/question_pairs.csv", stop=100000,
                 index_path="./index", models={}):
        algorithms = ["fasttext", "elmo", "bert"]
        self.models = {}
        for algorithm in algorithms:
            if algorithm in models:
                self.models[algorithm] = models[algorithm]
            else:
                self.models[algorithm] = None
        self.index_path = index_path
        self.docs = DocumentReader().get_docs(filepath, stop)
        self._save_orig_strings(self.docs)
        self._load_lemmatizer()
        self._preprocess_docs()
        self.load_algorithms()

    def _save_orig_strings(self, orig_strings):
        self.orig_strings = orig_strings
        with open(os.path.join(self.index_path, "orig_strings.pickle"), "wb") as pfile:
            pickle.dump(orig_strings, pfile)

    def _load_lemmatizer(self):
        self._m = pymystem3.mystem.Mystem()

    def _preprocess_string(self, instr):
        return [word for word in self._m.lemmatize(instr) if re.search(r'\w', word)]

    @timelogged("препроцессинг коллекции")
    def _preprocess_docs(self):
        self.docs = [self._preprocess_string(doc) for doc in self.docs]

    def load_algorithms(self):
        print("Started loading algorithms")
        self._algorithms = {
            "tfidf": TFIDF(),
            "bm25": BM25(),
            "fasttext": FastText(self.models["fasttext"]),
            "elmo": ELMo(self.models["elmo"]),
            "bert": BERT(self.models["bert"])
        }

    def index(self, models):
        model_docs = {
            "tfidf": self.docs,
            "bm25": self.docs,
            "fasttext": self.docs,
            "elmo": self.docs,
            "bert": self.orig_strings
        }
        models = [model for model in models
                  if self._algorithms[model].process(model_docs[model], index_path=self.index_path)]
        return models

class TFIDF:
    def __init__(self):
        try:
            self.vectorizer = TfidfVectorizer()
        except Exception as e:
            print("log")
        self.label = "tfidf"

    @timelogged("индексация TF-IDF")
    def process(self, docs, index_path):
        try:
            if os.path.exists(os.path.join(index_path, self.label+".pickle")):
                return True
            corpus = [" ".join(doc) for doc in docs]
            X = self.vectorizer.fit_transform(corpus)
            vocabulary = self.vectorizer.get_feature_names()
            index = csr_matrix(X, dtype='float64')
            struct = [vocabulary, index]
            with open(os.path.join(index_path, self.label+".pickle"), "wb") as pfile:
                pickle.dump(struct, pfile)
            return True
        except Exception as e:
            print("log")
            return False

class BM25:
    def __init__(self, k=2.0, b=0.75):
        self.label = "bm25"
        self.k = k
        self.b = b
        self.vocab = None
        self.docs = None
        self.index = None

    def preindex(self):
        """
        docs: list of lists
        """
        indptr = [0]
        indices = []
        data = []
        self.vocab = {}
        for d in self.docs:
                for term in d:
                        index = self.vocab.setdefault(term, len(self.vocab))
                        indices.append(index)
                        data.append(1)
                indptr.append(len(indices))
        self.index = csr_matrix((data, indices, indptr), dtype=int)

    def get_avgdl(self):
        ones = [[1] for _ in range(self.index.shape[1])]
        lens = np.dot(self.index, csr_matrix(ones)).toarray()
        self.avgdl = np.mean([g[0] for g in lens])

    def get_idf(self, word):
        n_qi = self.non_zero[self.vocab[word]]
        idf = log((self.N - n_qi + 0.5) /
                  (n_qi + 0.5))
        return idf

    def get_idf_struct(self):
        self.N = self.index.shape[0]
        self.non_zero = self.index.getnnz(0)
        self.idf_struct = {word: self.get_idf(word)
                           for word in sorted(self.vocab, key=self.vocab.get)}

    def tf(self, term, termlist):
        if not termlist:
            return 0
        return termlist.count(term)/len(termlist)

    def idf(self, word):
        return self.idf_struct[word]

    def get_bm25(self, word, D):
        IDF = self.idf(word)
        TF = self.tf(word, D)
        l = len(D)

        return IDF * TF * (self.k + 1) / \
               TF + self.k * (1 - self.b + self.b * (l / self.avgdl))

    @timelogged("индексация BM25")
    def process(self, docs, index_path):
        try:
            if os.path.exists(os.path.join(index_path, self.label+".pickle")):
                return True
            self.docs = docs
            self.preindex()
            self.get_avgdl()
            self.get_idf_struct()
            indptr = [0]
            indices = []
            data = []
            self.vocab = {}
            for d in docs:
                for term in d:
                    index = self.vocab.setdefault(term, len(self.vocab))
                    indices.append(index)
                    data.append(self.get_bm25(term, d))
                indptr.append(len(indices))
            self.index = csr_matrix((data, indices, indptr), dtype='float64')
            struct = [self.vocab, self.index]

            with open(os.path.join(index_path, self.label+".pickle"), "wb") as pfile:
                pickle.dump(struct, pfile)
            return True
        except Exception as e:
            print("log")
            return False

class FastText:
    def __init__(self, model=None):
        try:
            if model:
                self.model = model
            else:
                self.model = KeyedVectors.load("./models/fasttext/model.model")
        except Exception as e:
            print("log")
        self.label = "fasttext"

    @timelogged("индексация FastText")
    def process(self, docs, index_path):
        try:
            vec_size = self.model.vector_size
            index = np.zeros((len(docs), vec_size))
            index = index.reshape((len(docs), vec_size))
            for i, doc in enumerate(docs):
                entity = np.zeros(vec_size)
                for word in doc:
                    try:
                        entity += self.model[word]
                    except AttributeError:
                        pass
                if len(doc):
                    index[i] = entity / len(doc)
            struct = [{}, index]
            with open(os.path.join(index_path, self.label+".pickle"), "wb") as pfile:
                pickle.dump(struct, pfile)
            return True
        except Exception as e:
            print("log")
            return False

class ELMo:
    def __init__(self, model=None):
        self.label = "elmo"
        try:
            if model:
                self.model = model
            else:
                self.model = elmo_model()
        except:
            pass

    @timelogged("индексация ELMo")
    def process(self, docs, index_path):
        docs = docs
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                              gpu_options=gpu_options)) as sess:
            with tf.device('/gpu:0'):
                # It is necessary to initialize variables once before running inference.
                sess.run(tf.global_variables_initializer())

                start = time.time()
                vectors = []
                batch_size = 50
                bs, be = 0, batch_size

                while be <= len(docs):
                    batch = docs[bs:be]

                    elmo_vectors = get_elmo_vectors(
                        sess, batch, self.model.batcher, self.model.sentence_character_ids,
                        self.model.elmo_sentence_input)


                    # Due to batch processing, the above code produces for each sentence
                    # the same number of token vectors, equal to the length of the longest sentence
                    # (the 2nd dimension of the elmo_vector tensor).
                    # If a sentence is shorter, the vectors for non-existent words are filled with zeroes.
                    # Let's make a version without these redundant vectors:

                    cropped_vectors = []
                    for vect, sent in zip(elmo_vectors, batch):
                        cropped_vector = vect[:len(sent), :]
                        cropped_vectors.append(cropped_vector)

                    vectors += cropped_vectors
                    bs += batch_size
                    be += batch_size

            index = np.array([np.sum(v, axis=0) / len(v) for v in vectors])
            struct = [{}, index]
            with open(os.path.join(index_path, self.label+".pickle"), "wb") as pfile:
                pickle.dump(struct, pfile)
            return True


class BERT:
    def __init__(self, model=None):
        try:
            if model:
                self.model = model
            else:
                self.model = BertClient()
        except Exception as e:
            print("log")
        self.label = "bert"

    @timelogged("индексация BERT")
    def process(self, docs, index_path):
        try:
            nulls = [idx for idx, el in enumerate(docs) if not el]
            for idx in nulls:
                docs[idx] = "."
            index = self.model.encode(docs)
            with open(os.path.join(index_path, self.label+".pickle"), "wb") as pfile:
                pickle.dump(index, pfile)
            index = index.copy()
            for idx in nulls:
                index[idx] = np.array([0] * 768)
            struct = [{}, index]
            with open(os.path.join(index_path, self.label+".pickle"), "wb") as pfile:
                pickle.dump(struct, pfile)
            return True
        except Exception as e:
            print(e)
            return False
