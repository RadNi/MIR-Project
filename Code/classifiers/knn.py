from Code.classifiers.classifier import Classifier
from Code.indexer import Indexer
from Code.parser import EnglishParser
from Code.utils import create_vector_space, VectorSpace
import scipy.sparse.linalg
import numpy as np
import csv


class KNNClassifier(Classifier):

    def __init__(self, neighbor_count: int):
        super().__init__()
        self.neighbor_count = neighbor_count
        self.parser = EnglishParser(preload_corpus=False)
        self.indexer = Indexer("english", generic=True)
        self.indexer.parser = self.parser
        self.vector_space: VectorSpace = None
        self.doc_to_tag_map = {}

    def train(self, train_data, model_address=None, vs_address=None):
        self.parser.load_english_documents(train_data, is_data_tagged=True)
        if model_address is None:
            self.indexer.index(should_write=True, file_name="DataSet/indexes/english_train_index")
        else:
            self.indexer.index_filename = model_address
            self.indexer.read_index_table()
        self.vector_space = create_vector_space(self.indexer, table_file_saved=True, should_write=True,
                                                write_address="DataSet/vector_spaces/english_train_space",
                                                vs_address=vs_address)
        self._generate_doc_to_tag_map(train_data[1:])
        confusion_matrices = self._predict_and_calculate_confusion_matrix(train_data[1:])
        print("Training Finished")
        self.print_information(confusion_matrices)

    def test(self, test_data):
        confusion_matrices = self._predict_and_calculate_confusion_matrix(test_data[1:])
        print("Test Finished")
        self.print_information(confusion_matrices)

    def _generate_doc_to_tag_map(self, train_data):
        for i, row in enumerate(train_data):
            if row[0] not in self.tags_list:
                self.tags_list.append(row[0])
            self.doc_to_tag_map[i] = row[0]

    def predict_single_input(self, input_str):
        lemmatized_tokens = self.parser.parse_text(input_str, remove_del=True)
        doc_id_and_dist_list = []
        current_doc_vec = self.vector_space.calculate_query_vec(lemmatized_tokens, mode="ntn", sparse=False)
        for doc_id in self.vector_space.doc_dict.keys():
            other_doc_vec = self.vector_space.get_doc_vec(doc_id, mode="ntn", sparse=False)
            doc_id_and_dist_list.append((doc_id, np.linalg.norm(current_doc_vec - other_doc_vec)))
        doc_id_and_dist_list.sort(key=lambda x: x[1])
        count_per_tag = {}
        for tag in self.tags_list:
            count_per_tag[tag] = 0
        break_index = min(self.neighbor_count, len(doc_id_and_dist_list))
        for pair in doc_id_and_dist_list[:break_index]:
            count_per_tag[self.doc_to_tag_map[pair[0]]] += 1
        max_tag = self.tags_list[0]
        max_count = -1
        for tag in self.tags_list:
            if count_per_tag[tag] > max_count:
                max_tag = tag
                max_count = count_per_tag[tag]
        print(f'predicted {input_str} to {max_tag}')
        return max_tag


if __name__ == '__main__':
    classifier = KNNClassifier(5)
    with open("DataSet/training_data/phase2_train.csv") as f:
        classifier.train(list(csv.reader(f, delimiter=',')), model_address="DataSet/indexes/english_train_index",
                         vs_address="DataSet/vector_spaces/english_train_space")
