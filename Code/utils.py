import numpy as np
from Code.indexer import Indexer
from Code.constants import *
import scipy.sparse
import scipy.sparse.linalg
import math


class VectorSpace:

    def __init__(self, postings_index: dict, total_doc_count: int = 0, model: dict = {}):
        self.postings = postings_index
        self.word2index = list(postings_index.keys())
        self.word_count = len(self.word2index)

        if len(model.keys()) == 0:
            self.doc_count = total_doc_count
            self.doc_dict = {}
        else:
            self.doc_count = len(model.keys())
            self.doc_dict = model

    def add_doc_vec(self, doc_id, word_set):
        self.doc_dict[doc_id] = set(word_set)

    def get_doc_vec(self, doc_id, mode="ltc", sparse=True):
        word_list = self.doc_dict[doc_id]
        if sparse:
            new_vec = scipy.sparse.lil_matrix((self.word_count, 1))
            for word in word_list:
                if word not in self.postings.keys():
                    continue
                posting: dict = self.postings[word]
                if mode[0] == 'l':
                    tf = 1 + math.log2(len(posting[doc_id]))
                elif mode[0] == 'n':
                    tf = len(posting[doc_id])
                else:
                    tf = 0.001
                if mode[1] == 't':
                    idf = math.log2((self.doc_count + 1) / len(posting.keys()))
                else:
                    idf = 0.001
                new_vec[self.word2index.index(word), 0] = tf * idf
            if mode[2] == 'c':
                return self._normalize(new_vec)
            elif mode[2] == 'n':
                return new_vec
            else:
                return new_vec
        else:
            new_vec = np.zeros((self.word_count, 1))
            for word in word_list:
                if word not in self.postings.keys():
                    continue
                posting: dict = self.postings[word]
                if mode[0] == 'l':
                    tf = 1 + math.log2(len(posting[doc_id]))
                elif mode[0] == 'n':
                    tf = len(posting[doc_id])
                else:
                    tf = 0.001
                if mode[1] == 't':
                    idf = math.log2((self.doc_count + 1) / len(posting.keys()))
                else:
                    idf = 0.001
                new_vec[self.word2index.index(word), 0] = tf * idf
            if mode[2] == 'c':
                return self._normalize(new_vec)
            elif mode[2] == 'n':
                return new_vec
            else:
                return new_vec

    def write_vec_to_file(self, mode, write_address):
        if write_address is None:
            filename = "DataSet/vector_spaces/" + mode + "_vector_space"
        else:
            filename = write_address
        with open(filename, 'w', encoding="utf8") as f:
            f.write(str(self.doc_dict))

    def calculate_query_vec(self, query_words_list: list, mode="lnc", sparse=True):
        words_set = set(query_words_list)
        if sparse:
            vec = scipy.sparse.lil_matrix((self.word_count, 1))
            for word in words_set:
                if mode[1] == 't' and word not in self.postings.keys():
                    continue
                count = query_words_list.count(word)
                if mode[0] == 'l':
                    tf = 1 + math.log2(count)
                elif mode[0] == 'n':
                    tf = count
                else:
                    tf = 0.001
                if mode[1] == 'n':
                    idf = 1
                elif mode[1] == 't':
                    idf = math.log2((self.doc_count + 1) / len(self.postings[word].keys()))
                else:
                    idf = 0.001
                vec[self.word2index.index(word), 0] = tf * idf
            if mode[2] == 'c':
                return self._normalize(vec)
            elif mode[2] == 'n':
                return vec
            else:
                return vec
        else:
            vec = np.zeros((self.word_count, 1))
            for word in words_set:
                if mode[1] == 't' and word not in self.postings.keys():
                    continue
                count = query_words_list.count(word)
                if mode[0] == 'l':
                    tf = 1 + math.log2(count)
                elif mode[0] == 'n':
                    tf = count
                else:
                    tf = 0.001
                if mode[1] == 'n':
                    idf = 1
                elif mode[1] == 't':
                    idf = math.log2((self.doc_count + 1) / len(self.postings[word].keys()))
                else:
                    idf = 0.001
                vec[self.word2index.index(word), 0] = tf * idf
            if mode[2] == 'c':
                return self._normalize(vec)
            elif mode[2] == 'n':
                return vec
            else:
                return vec

    @staticmethod
    def read_vector_space_model(mode, address=None):
        if address is None:
            filename = "DataSet/vector_spaces/" + mode + "_vector_space"
        else:
            filename = address
        with open(filename, "r", encoding="utf8") as f:
            return eval(f.read())

    @staticmethod
    def _normalize(vec: np.array):
        return vec / scipy.sparse.linalg.norm(vec)


def create_vector_space(indexer=None, table_file_saved=True, vs_address=None, should_write=True, write_address=None):
    if indexer is None:
        indexer = Indexer(MODE)
    if table_file_saved:
        index_table = indexer.read_index_table()
        print(index_table)
    else:
        index_table = indexer.posting_list
    # exit(1)
    vs = VectorSpace(index_table, len(indexer.parser.get_docids()))
    if vs_address is None:
        i = 1
        print(len(index_table))
        # exit(0)
        for id in indexer.parser.get_docids():
            print(id, ':', i, '/', len(indexer.parser.get_docids()))
            words = set(indexer.index_single_doc(id).keys())
            vs.add_doc_vec(id, words)
            i += 1
        if should_write:
            vs.write_vec_to_file(MODE, write_address)
        print(vs.doc_dict)
    else:
        vs.doc_dict = VectorSpace.read_vector_space_model(MODE, vs_address)
    return vs


if __name__ == '__main__':
    create_vector_space()
    VectorSpace.read_vector_space_model(MODE)
