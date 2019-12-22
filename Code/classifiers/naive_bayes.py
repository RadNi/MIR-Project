import math
import numpy as np

from Code.classifiers.classifier import Classifier
from Code.parser import EnglishParser


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
        tags, data = train_data
        self.english_parser.load_english_documents(data, is_data_tagged=False, has_header=False)
        self._create_class_index(tags)
        confusion_matrices = self._predict_and_calculate_confusion_matrix(train_data)
        print("Training Finished")
        self.print_information(confusion_matrices)

    def test(self, test_data):
        confusion_matrices = self._predict_and_calculate_confusion_matrix(test_data)
        print("Test Finished")
        self.print_information(confusion_matrices)

    def predict_single_input(self, input_str):
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
                else:
                    current_prob += math.log2(1 / (self.class_frequencies[tag] + self.all_words_count))
            if current_prob > best_prob:
                best_prob = current_prob
                best_tag = tag
        return best_tag

    def _create_class_index(self, tag_list):
        for doc_id in range(len(tag_list)):
            lemmatized_tokens = self.english_parser.parse_doc(doc_id, remove_del=True)
            current_tag = tag_list[doc_id]
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
        for i in range(len(tag_list)):
            self.class_probabilities[tag_list[i]] += 1
        for class_tag in self.tags_list:
            self.class_probabilities[class_tag] /= len(tag_list) - 1
