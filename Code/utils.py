import numpy as np
from Code.indexer import Indexer
from Code.constants import *
import scipy.sparse
import scipy.sparse.linalg
import math


class VectorSpace:

    def __init__(self, postings_index: dict, total_doc_count: int):
        self.postings = postings_index
        self.doc_count = total_doc_count
        self.word2index = list(postings_index.keys())
        self.doc_dict = {}
        self.word_count = len(self.word2index)

    def add_doc_vec(self, doc_id, word_set):
        self.doc_dict[doc_id] = set(word_set)

    def get_doc_vec(self, doc_id):
        new_vec = scipy.sparse.lil_matrix((self.word_count, 1))
        word_list = self.doc_dict[doc_id]
        for word in word_list:
            posting: dict = self.postings[word]
            idf = math.log2((self.doc_count + 1) / len(posting.keys()))
            tf = 1 + math.log2(len(posting[doc_id]))
            new_vec[self.word2index.index(word), 0] = tf * idf
        return self._normalize(new_vec)

    def write_vec_to_file(self, mode):
        filename = "DataSet/vector_spaces/" + mode + "_vector_space"
        with open(filename, 'w', encoding="utf8") as f:
            f.write(str(self.doc_dict))

    def calculate_query_vec(self, query_words_list: list):
        vec = scipy.sparse.lil_matrix((self.word_count, 1))
        words_set = set(query_words_list)
        for word in words_set:
            count = query_words_list.count(word)
            idf = 1
            tf = 1 + math.log2(count)
            vec[self.word2index.index(word), 0] = tf * idf
        return self._normalize(vec)

    @staticmethod
    def read_vector_space_model(mode):
        filename = "DataSet/vector_spaces/" + mode + "_vector_space"
        with open(filename, "r", encoding="utf8") as f:
            return eval(f.read())

    @staticmethod
    def _normalize(vec: np.array):
        return vec / scipy.sparse.linalg.norm(vec)


def create_vector_space():
    indexer = Indexer(MODE)
    index_table = indexer.read_index_table()
    print(index_table)
    # exit(1)
    vs = VectorSpace(index_table, len(indexer.parser.get_docids()))
    i = 1
    print(len(index_table))
    # exit(0)
    for id in indexer.parser.get_docids():
        print(id, ':', i, '/', len(indexer.parser.get_docids()))
        words = set(indexer.index_single_doc(id).keys())
        vs.add_doc_vec(id, words)
        i += 1
    vs.write_vec_to_file(MODE)
    print(vs.doc_dict)


if __name__ == '__main__':
    create_vector_space()
    VectorSpace.read_vector_space_model(MODE)
