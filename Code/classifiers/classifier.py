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

    def _predict_and_calculate_confusion_matrix(self, predict_dataset):
        # TP FP FN TN
        confusion_matrices = {}
        for tag in self.tags_list:
            confusion_matrices[tag] = [0, 0, 0, 0]
        for data in predict_dataset:
            correct_tag = data[0]
            predicted_tag = self.predict_single_input(f'{data[1]} {data[2]}')
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
