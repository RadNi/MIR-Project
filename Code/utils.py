import numpy as np
from Code.indexer import Indexer
from Code.parser import *
from Code.constants import *


class VectorSpace:

    def __init__(self, postings_index: dict, total_doc_count: int):
        self.postings = postings_index
        self.doc_count = total_doc_count
        self.word2index = list(postings_index.keys())
        self.doc_dict = {}
        self.word_count = len(self.word2index)

    def add_doc_vec(self, doc_id, doc_word_set):
        new_vec = np.zeros((self.word_count,))
        for word in doc_word_set:
            posting: dict = self.postings[word]
            idf = np.log2((self.doc_count + 1) / len(posting.keys()))
            tf = 1 + np.log2(len(posting[doc_id]))
            new_vec[self.word2index.index(word)] = tf * idf
        self.doc_dict[doc_id] = self._normalize(new_vec)

    def write_vec_to_file(self, filename):
        np.save(filename, self.doc_dict)
        # with open(filename, 'w') as f:
        #     f.write(str(self.doc_dict))

    def calculate_query_vec(self, query_words_list: list):
        vec = np.zeros((self.word_count,))
        words_set = set(query_words_list)
        for word in words_set:
            count = query_words_list.count(word)
            idf = 1
            tf = 1 + np.log2(count)
            vec[self.word2index.index(word)] = tf * idf
        return self._normalize(vec)

    @staticmethod
    def _normalize(vec: np.array):
        return vec / np.linalg.norm(vec)


if __name__ == '__main__':
    indexer = Indexer(MODE)
    index_table = indexer.read_index_table()
    vs = VectorSpace(index_table, len(indexer.parser.get_docids()))
    i = 1
    print(len(index_table))
    # exit(0)
    for id in indexer.parser.get_docids()[:10]:
        print(id, ':', i, '/', len(indexer.parser.get_docids()))
        words = indexer.index_single_doc(id).keys()
        vs.add_doc_vec(id, words)
        i += 1
    vs.write_vec_to_file(str(MODE) + "_vector_space")
    print(vs.doc_dict)