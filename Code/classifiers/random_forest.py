from Code.classifiers.classifier import Classifier


class RandomForestClassifier(Classifier):

    def __init__(self):
        super().__init__()

    def train(self, train_data):
        return super().train(train_data)

    def test(self, test_data):
        return super().test(test_data)
