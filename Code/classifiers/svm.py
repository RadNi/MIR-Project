from Code.classifiers.classifier import Classifier


class SVMClassifier(Classifier):

    def __init__(self, c: float):
        super().__init__()
        self.c = c

    def train(self, train_data):
        return super().train(train_data)

    def test(self, test_data):
        return super().test(test_data)
