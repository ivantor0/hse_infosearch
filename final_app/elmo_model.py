import tensorflow as tf

from timedlogger import timelogged
from elmo_helpers import load_elmo_embeddings


class elmo_model:

    @timelogged("загрузка модели ELMo")
    def __init__(self, elmo_path='./models/elmo'):
        tf.reset_default_graph()
        self.batcher, self.sentence_character_ids, self.elmo_sentence_input = \
            load_elmo_embeddings(elmo_path)
