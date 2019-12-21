from sklearn import svm
from Code.classifiers.SKClassifier import SKClassifier


class SVMClassifier(SKClassifier):
    def __init__(self, penalty):
        super().__init__()
        self.model = svm.SVC(kernel="linear", C=penalty)


if __name__ == '__main__':
    for penalty in [1/2, 1, 3/2, 2]:
        svmc = SVMClassifier(penalty=penalty)
        train_set = svmc.read_data_from_file("DataSet/phase2/phase2_train.csv")
        svmc.train(train_set)
        svmc.test(svmc.read_data_from_file("DataSet/phase2/phase2_test.csv"))
