import numpy as np

from Code.classifiers.classifier import Classifier
from sklearn import metrics
# from sklearn.ensemble import RandomForestClassifier as sklearnRandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


class SKClassifier(Classifier):

    def __init__(self, class_name):
        super().__init__(class_name)
        self.model = None
        # self.model = sklearnRandomForestClassifier(max_depth=2000, random_state=0)
        self.train_set = []
        self.test_set = []
        self.train_set_vs = []
        self.test_set_vs = []
        self.test_set_labels = []
        self.vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
        self.train_set_labels = []

    def train(self, train_data):
        self.train_set_labels = train_data[1]
        self.tags_list = set(self.train_set_labels)
        self.train_set = train_data[0]
        self.train_set_vs = self.vectorizer.fit_transform(self.train_set)

        self.model.fit(self.train_set_vs, self.train_set_labels)
        # print(self.model.feature_importances_)
        # train_test_split()

    def predict_and_show_metrics(self, correct_tags, data=None, vector_space: np.ndarray = None):
        if data is not None:
            predictions = np.ndarray(shape=(len(data),), dtype=np.int32)
            for i, row in enumerate(data):
                predictions[i] = self.predict_single_input(input_str=row)
        else:
            predictions = np.ndarray(shape=(len(vector_space.shape[0]),), dtype=np.int32)
            for i in range(vector_space.shape[0]):
                vec = vector_space[i, :]
                predictions[i] = self.predict_single_input(input_vec=vec)
        self.print_metrics(predictions, correct_tags)

    def test(self, test_data):

        # self.test_set_labels = test_data[1]
        self.test_set = test_data
        self.test_set_vs = self.vectorizer.transform(self.test_set)

        y_pred = self.model.predict(self.test_set_vs)
        return y_pred

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


# if __name__ == '__main__':
#     print(SKClassifier().read_data_from_file("DataSet/phase2/phase2_train.csv"))
#
# if __name__ == '__main__':
#     rf = RandomForestClassifier()
#     train_set = rf.read_data_from_file("DataSet/phase2/phase2_train.csv")
#     rf.train(train_set)
#     rf.test(rf.read_data_from_file("DataSet/phase2/phase2_test.csv"))
