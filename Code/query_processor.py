import editdistance
from Code.constants import MIN_IOU, FARSI
import hazm
import nltk


class QueryCorrector:
    def __init__(self, bi_gram_index: list):
        self.index_list: list = bi_gram_index
        # indexes are lists of tuples

    def correct_query(self, query_string: str):
        if FARSI:
            correct_words = []
            words = hazm.word_tokenize(query_string)
            for word in words:
                correct_words.append(self.correct_word(word))
            return " ".join(correct_words)
        else:
            correct_words = []
            words = nltk.tokenize.word_tokenize(query_string)
            for word in words:
                correct_words.append(self.correct_word(word))
            return " ".join(correct_words)

    def correct_word(self, word: str):
        grams: set = self.extract_bi_grams(word)
        word_set = set()
        for gram in grams:
            for item in self.index_list:
                if item[0] == gram:
                    word_set.update(set(item[1]))
                    break
        # Calculating IoU
        final_set = []
        for word in word_set:
            word_grams = self.extract_bi_grams(word)
            if self.calculate_iou(grams, word_grams) > MIN_IOU:
                final_set.append(word)
        final_set.sort(key=lambda x: editdistance.distance(x, word))
        return final_set[0]

    @staticmethod
    def extract_bi_grams(word: str):
        output = set()
        for i in range(len(word) - 1):
            output.add(word[i:i + 2])
        return output

    @staticmethod
    def calculate_iou(first_set: set, second_set: set):
        return len(first_set.intersection(second_set)) / len(first_set.union(second_set))
