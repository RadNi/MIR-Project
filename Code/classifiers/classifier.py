class Classifier:

    def __init__(self):
        self.cluster_count = 4
        self.tags_list = []

    def train(self, train_data):
        return NotImplementedError

    def test(self, test_data):
        return NotImplementedError

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
