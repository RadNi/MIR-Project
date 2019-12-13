import numpy as np


class Classifier:

    def __init__(self):
        self.cluster_count = 4

    def train(self, train_data):
        return NotImplementedError

    def test(self, test_data):
        return NotImplementedError


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

    def train(self, train_data):
        return super().train(train_data)

    def test(self, test_data):
        return super().test(test_data)


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

