import time

from sklearn.feature_extraction.text import TfidfVectorizer

from Code.classifiers.SKClassifier import SKClassifier
from Code.indexer import Indexer
from Code.parser import EnglishParser
from Code.utils import create_vector_space, VectorSpace
import scipy.sparse.linalg
import numpy as np
import csv


class KNNClassifier(SKClassifier):

    def __init__(self, neighbor_count: int):
        super().__init__()
        self.neighbor_count = neighbor_count
        self.full_train_dot_vec = np.ndarray((1,))

    def train(self, train_data, should_eval=True):
        self.train_set_labels = train_data[1]
        self.tags_list = list(set(self.train_set_labels))
        self.train_set = train_data[0]
        self.train_set_vs = self.vectorizer.fit_transform(self.train_set)
        tmp = self.train_set_vs.dot(self.train_set_vs.T)
        self.full_train_dot_vec = np.diag(tmp.toarray())[:, np.newaxis]
        print("Training Finished")
        if should_eval:
            self.predict_and_show_metrics(self.train_set_labels, self.train_set)

    def test(self, test_data):
        self.test_set_labels = test_data[1]
        self.test_set = test_data[0]
        self.test_set_vs = self.vectorizer.transform(self.test_set)
        print("Test Finished")
        self.predict_and_show_metrics(self.test_set_labels, self.test_set)

    def predict_single_input(self, input_str):
        current_doc_vec = self.vectorizer.transform([input_str])
        now = time.time_ns()

        train_to_current_dot_vec = self.train_set_vs.dot(current_doc_vec.T).toarray()
        current_vec_dot = current_doc_vec.dot(current_doc_vec.T).toarray()
        distances_vector = (current_vec_dot - 2 * train_to_current_dot_vec + self.full_train_dot_vec).ravel()

        #     other_doc_vec = self.train_set_vs[doc_id]
        #     doc_id_and_dist_list.append((doc_id,
        #                                  scipy.sparse.csr_matrix.dot(current_doc_vec, current_doc_vec.transpose()) -
        #                                  2 * scipy.sparse.csr_matrix.dot(current_doc_vec, other_doc_vec.transpose()) +
        #                                 scipy.sparse.csr_matrix.dot(other_doc_vec, other_doc_vec.transpose())))
        args = np.argsort(distances_vector)
        count_per_tag = {}
        min_dist_for_tag = {}
        for tag in self.tags_list:
            count_per_tag[tag] = 0
            min_dist_for_tag[tag] = 0
        break_index = min(self.neighbor_count, len(args))
        for index in args[:break_index]:
            count_per_tag[self.train_set_labels[index]] += 1
            if min_dist_for_tag[self.train_set_labels[index]] > distances_vector[index]:
                min_dist_for_tag[self.train_set_labels[index]] += distances_vector[index]
        max_count = max(count_per_tag.values())
        max_tag = -1
        min_dist = np.inf
        for tag in self.tags_list:
            if count_per_tag[tag] >= max_count and min_dist_for_tag[tag] < min_dist:
                max_tag = tag
                min_dist = min_dist_for_tag[tag]

        print(max_tag, time.time_ns() - now)
        return max_tag


if __name__ == '__main__':
    classifier = KNNClassifier(5)
    # with open("DataSet/phase2/phase2_train.csv") as f:
    #     classifier.train(list(csv.reader(f, delimiter=',')))
    classifier.train(classifier.read_data_from_file("DataSet/phase2/phase2_train.csv"), should_eval=True)
    classifier.test(classifier.read_data_from_file("DataSet/phase2/phase2_test.csv"))
