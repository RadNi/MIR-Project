from Code.classifiers.classifier import Classifier


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
