from sklearn import metrics
import numpy as np
from Code.indexer import Indexer
from Code.utils import create_vector_space, VectorSpace
from Code.constants import FILE_ADDR_PREFIX

from Code.indexer import Indexer, csv


class Classifier:

    def __init__(self, class_name, **kwargs):
        self.class_name = class_name
        self.cluster_count = 4
        self.tags_list = []
        self.indexer = Indexer("english", is_data_tagged=True, preload_corpus=False, **kwargs)

    @staticmethod
    def _calculate_all_words(indexer):
        print("Calculating data set words.")
        result = []
        line_count = 0
        for docid in indexer.parser.get_docids():
            line_count += 1
            if line_count % 1000 == 0:
                print(f'\tProcessed {line_count} doc.')

            result.append(" ".join(term for term in list(indexer.create_all_terms(docid))))
        return result

    def train(self, train_data):
        return NotImplementedError

    def test(self, test_data):
        return NotImplementedError

    def read_data_from_file(self, file_name):
        self.indexer = Indexer("english", is_data_tagged=True, preload_corpus=False)
        self.indexer.set_adresses(
            index_address="DataSet/phase2/index_train",
            bigram_addres="DataSet/phase2/bigram_train.csv")
        self.indexer.parser.read_english_documents(filename=file_name, set_docs=True)
        labels = [int(label)for label in self.indexer.parser.first_row]
        return self._calculate_all_words(self.indexer), labels

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

    def print_metrics(self, predictions, correct_tags):
        print("Confusion Matrix:\n", metrics.confusion_matrix(correct_tags, predictions))
        # TODO here we must do something with our tags_list and generic confusion matrix format
        # self.print_information(metrics.confusion_matrix(correct_tags, predictions))
        print(metrics.classification_report(correct_tags, predictions))
        print(metrics.accuracy_score(correct_tags, predictions))
        print('Mean Absolute Error:', metrics.mean_absolute_error(correct_tags, predictions))
        print('Mean Squared Error:', metrics.mean_squared_error(correct_tags, predictions))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(correct_tags, predictions)))

    def _predict_and_calculate_confusion_matrix(self, predict_dataset, has_header=True):
        # TP FP FN TN
        confusion_matrices = {}
        if not has_header:
            dataset, labels = predict_dataset
            predict_dataset = zip(labels, dataset)
        for tag in self.tags_list:
            confusion_matrices[tag] = [0, 0, 0, 0]
        for data in predict_dataset:
            correct_tag = data[0]
            if has_header:
                predicted_tag = self.predict_single_input(f'{data[1]} {data[2]}')
            else:
                predicted_tag = self.predict_single_input(data[1])

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

    def predict_single_input(self, input_str):
        return NotImplementedError

    def get_corpus_address(self):
        return FILE_ADDR_PREFIX + "/" + self.class_name + "/phase1_labeled_"

    def rewrite_csv_with_label(self, source_file, labels):
        documents = {1: [], 2: [], 3: [], 4: []}
        with open(source_file, encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    documents[labels[line_count - 1]].append([row[0], row[1]])
                line_count += 1
            print(f'Processed {line_count} lines.')

        for label in list(documents.keys()):
            with open(self.get_corpus_address() + str(label) + ".csv", mode='w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(["Title", "Text"])
                for doc in documents[label]:
                    csv_writer.writerow(doc)

    @staticmethod
    def show_prediction_result(y_pred, y_real):

        print("Confusion Matrix:\n", metrics.confusion_matrix(y_real, y_pred))
        # TODO here we must do something with our tags_list and generic confusion matrix format
        # self.print_information(metrics.confusion_matrix(self.test_set_labels, y_pred))
        print(metrics.classification_report(y_real, y_pred))
        print(metrics.accuracy_score(y_real, y_pred))
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_real, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_real, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_real, y_pred)))

    def customised_index_creator(self, label):
        indexer = Indexer("english", preload_corpus=False, is_data_tagged=False)
        indexer.parser.read_english_documents(self.get_corpus_address() + str(label) + ".csv", True)
        indexer.bigram_index_filename = self.get_corpus_address() + str(label) + "_bigram"
        indexer.index_filename = self.get_corpus_address() + str(label) + "_index"
        return indexer

    def create_index_and_bigram(self):
        for i in range(1, 5):
            indexer = self.customised_index_creator(i)
            indexer.index()
            indexer.create_bigram_index()

    def create_vector_space(self):
        for i in range(1, 5):
            indexer = self.customised_index_creator(i)
            create_vector_space(indexer=indexer, write_address=self.get_corpus_address() + str(i) + "_vs",
                                should_write=True)

    def read_vector_space_model(self, label):
        print(self.get_corpus_address() + str(label) + "_vs")
        return VectorSpace.read_vector_space_model("english", self.get_corpus_address() + str(label) + "_vs")
