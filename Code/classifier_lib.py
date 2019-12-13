import math

import numpy as np
from Code.parser import EnglishParser
from Code.indexer import Indexer


class Classifier:

    def __init__(self):
        self.cluster_count = 4
        self.tags_list = []

    def train(self, train_data):
        return NotImplementedError

    def test(self, test_data):
        return NotImplementedError

    def print_information(self, confusion_matrices):
        for tag in self.tags_list:
            cm = confusion_matrices[tag]
            precision = cm[0] / (cm[0] + cm[1])
            recall = cm[0] / (cm[0] + cm[2])
            accuracy = (cm[0] + cm[3]) / (sum(cm))
            f1 = 2 * precision * recall / (precision + recall)
            print(f"\tFor class {tag}:")
            print(f"\t\tAccuracy: {accuracy}")
            print(f"\t\tPrecision: {precision}")
            print(f"\t\tRecall: {recall}")
            print(f"\t\tF1: {f1}")
            print("-------------------------------------------")


class KNNClassifier(Classifier):

    def __init__(self, neighbor_num: int, max_iter: int = 100, epsilon: float = 0.001):
        super().__init__()
        self.neighbor_num = neighbor_num
        self.max_iter = max_iter
        self.epsilon = epsilon

    def train(self, train_data):
        return super().train(train_data)

    def test(self, test_data):
        return super().test(test_data)


class NaiveBayesClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.english_parser = EnglishParser(preload_corpus=False)
        self.class_index = {}
        self.class_probabilities = {}
        self.class_frequencies = {}
        self.all_words_set = set()
        self.all_words_count = 0

    def train(self, train_data):
        self.english_parser.load_english_documents(train_data, is_data_tagged=True)
        self._create_class_index(train_data)
        confusion_matrices = self._predict_and_calculate_confusion_matrix(train_data[1:])
        print("Training Finished")
        self.print_information(confusion_matrices)

    def test(self, test_data):
        confusion_matrices = self._predict_and_calculate_confusion_matrix(test_data[1:])
        print("Test Finished")
        self.print_information(confusion_matrices)

    def _predict_and_calculate_confusion_matrix(self, predict_dataset):
        # TP FP FN TN
        confusion_matrices = {}
        for tag in self.tags_list:
            confusion_matrices[tag] = [0, 0, 0, 0]
        for data in predict_dataset:
            correct_tag = data[0]
            predicted_tag = self._predict_single_input(f'{data[1]} {data[2]}')
            for tag in self.tags_list:
                if tag == correct_tag:
                    if tag == predicted_tag:
                        confusion_matrices[tag][0] += 1
                    else:
                        confusion_matrices[tag][2] += 1
                else:
                    if tag == predicted_tag:
                        confusion_matrices[tag][1] += 1
                    else:
                        confusion_matrices[tag][3] += 1
        return confusion_matrices

    def _predict_single_input(self, input_str):
        lemmatized_tokens = self.english_parser.parse_text(input_str, remove_del=True)
        best_prob = -np.inf
        best_tag = self.tags_list[0]
        for tag in self.tags_list:
            current_prob = math.log2(self.class_probabilities[tag])
            current_index: dict = self.class_index[tag]
            for token in lemmatized_tokens:
                if token in current_index.keys():
                    current_prob += math.log2(
                        (current_index[token] + 1) / (self.class_frequencies[tag] + self.all_words_count))
            if current_prob > best_prob:
                best_prob = current_prob
                best_tag = tag
        return best_tag

    def _create_class_index(self, train_data):
        for doc_id in range(1, len(train_data)):
            lemmatized_tokens = self.english_parser.parse_doc(doc_id, remove_del=True)
            current_tag = train_data[doc_id][0]
            index_dict = {}
            if current_tag in self.class_index.keys():
                index_dict = self.class_index[current_tag]
            for token in lemmatized_tokens:
                if token not in self.all_words_set:
                    self.all_words_set.add(token)
                    self.all_words_count += 1
                if token in index_dict.keys():
                    index_dict[token] += 1
                else:
                    index_dict[token] = 1
            if current_tag not in self.class_index.keys():
                self.class_index[current_tag] = index_dict
        # Calculating sum(T_ct')
        for class_tag in self.class_index.keys():
            self.tags_list.append(class_tag)
            self.class_frequencies[class_tag] = sum(self.class_index[class_tag].values())

        # Calculating P(c)
        for class_tag in self.tags_list:
            self.class_probabilities[class_tag] = 0
        for i in range(1, len(train_data)):
            self.class_probabilities[train_data[i][0]] += 1
        for class_tag in self.tags_list:
            self.class_probabilities[class_tag] /= len(train_data) - 1


class SVMClassifier(Classifier):

    def __init__(self, c: float):
        super().__init__()
        self.c = c

    def train(self, train_data):
        return super().train(train_data)

    def test(self, test_data):
        return super().test(test_data)


class RandomForestClassifier(Classifier):

    def __init__(self):
        super().__init__()

    def train(self, train_data):
        return super().train(train_data)

    def test(self, test_data):
        return super().test(test_data)
