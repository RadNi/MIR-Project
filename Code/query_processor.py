import editdistance
from Code.constants import MIN_IOU
from Code.parser import Parser
import hazm
import nltk


class QueryCorrector:
    def __init__(self, bi_gram_dict: dict):
        self.grad_dict: dict = bi_gram_dict
        # indexes are lists of tuples

    def correct_query(self, query_string: str, mode):
        if mode == "persian":
            correct_words = []
            words = hazm.word_tokenize(query_string)
            for word in words:
                try:
                    correct_words.append(self.correct_word(word))
                except IndexError:
                    print(f"Word cannot be corrected: {word}")
                    correct_words.append(word)
            return " ".join(correct_words)
        else:
            correct_words = []
            words = nltk.tokenize.word_tokenize(query_string)
            for word in words:
                try:
                    correct_words.append(self.correct_word(word))
                except IndexError:
                    print(f"Word cannot be corrected: {word}")
                    correct_words.append(word)
            return " ".join(correct_words)

    def correct_word(self, the_word: str):
        grams: set = self.extract_bi_grams(the_word)
        word_set = set()
        for gram in grams:
            try:
                word_set.update(set(self.grad_dict[gram]))
            except KeyError:
                try:
                    word_set.update(set(self.grad_dict[gram.lower()]))
                except KeyError:
                    continue
        # Calculating IoU
        final_set = []
        for word in word_set:
            word_grams = self.extract_bi_grams(word)
            if self.calculate_iou(grams, word_grams) > MIN_IOU:
                final_set.append(word)
        final_set.sort(key=lambda x: editdistance.distance(x, the_word))
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
