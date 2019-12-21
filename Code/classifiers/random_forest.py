import numpy as np

from Code.classifiers.classifier import Classifier
from Code.indexer import Indexer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as sklearnRandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


class RandomForestClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.indexer = Indexer("english", is_data_tagged=True, preload_corpus=False)
        self.model = sklearnRandomForestClassifier(max_depth=2000, random_state=0)
        self.train_set = []
        self.test_set = []
        self.train_set_vs = []
        self.test_set_vs = []
        self.test_set_labels = []
        self.vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
        self.train_set_labels = []

    def read_data_from_file(self, file_name):
        self.indexer = Indexer("english", is_data_tagged=True, preload_corpus=False)
        self.indexer.set_adresses(
            index_address="DataSet/phase2/index_train",
            bigram_addres="DataSet/phase2/bigram_train.csv")
        self.indexer.parser.read_english_documents(filename=file_name, set_docs=True)
        labels = [int(label)for label in self.indexer.parser.first_row]
        return self._calculate_all_words(self.indexer), labels

    def train(self, train_data):
        self.train_set_labels = train_data[1]
        self.tags_list = set(self.train_set_labels)
        self.train_set = train_data[0]
        self.train_set_vs = self.vectorizer.fit_transform(self.train_set)

        self.model.fit(self.train_set_vs, self.train_set_labels)
        print(self.model.feature_importances_)
        # train_test_split()

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

    def test(self, test_data):

        self.test_set_labels = test_data[1]
        self.test_set = test_data[0]
        self.test_set_vs = self.vectorizer.transform(self.test_set)

        y_pred = self.model.predict(self.test_set_vs)

        print("Confusion Matrix:\n", metrics.confusion_matrix(self.test_set_labels, y_pred))
        # TODO here we must do something with our tags_list and generic confusion matrix format
        # self.print_information(metrics.confusion_matrix(self.test_set_labels, y_pred))
        print(metrics.classification_report(self.test_set_labels, y_pred))
        print(metrics.accuracy_score(self.test_set_labels, y_pred))
        print('Mean Absolute Error:', metrics.mean_absolute_error(self.test_set_labels, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(self.test_set_labels, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(self.test_set_labels, y_pred)))


if __name__ == '__main__':
    rf = RandomForestClassifier()
    train_set = rf.read_data_from_file("DataSet/phase2/phase2_train.csv")
    rf.train(train_set)
    rf.test(rf.read_data_from_file("DataSet/phase2/phase2_test.csv"))
