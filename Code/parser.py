from bs4 import BeautifulSoup
from hazm import *
from collections import Counter
import csv
import nltk.stem


class Parser:
    """
    Shall we ignore nim-fasele?
    What's the policy?
    """

    def __init__(self, freq_threshold, common_words_filename):
        self.common_words_filename = common_words_filename
        self.common_words = self._read_common_words(common_words_filename)
        self.documents = []
        self.freq_threshold = freq_threshold

    @staticmethod
    def _is_english(ch):
        return 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' or '0' <= ch <= '9'

    def _prepare_complete_text(self, doc):
        pass

    def _rm_delimiters(self, document):
        res = list(document)
        for i in range(len(document)):
            if self._is_delimiter(document[i]):
                res[i] = " "
        return "".join(c for c in res)

    def _is_delimiter(self, doc):
        pass

    def _normalize_doc(self, doc):
        return self.normalizer("".join(doc))

    def _tokenize(self, doc):
        return self.word_tokenize(doc)

    def _lemmatize_tokens(self, tokens):
        res = []
        for t in tokens:
            res.append(self.lemmatizer(t))

        return res

    def _read_common_words(self, filename):
        with open(filename, 'r', encoding="utf8") as f:
            return eval(f.read())

    def _prepare_text(self, text, remove_del=False, verbose=False, only_tokenize=False):
        if remove_del:
            text = self._rm_delimiters(text)
        if only_tokenize:
            return self._tokenize(text)
        doc_nr = self._normalize_doc(text)
        if verbose:
            print("Normalized", doc_nr)
        tokens = self._tokenize(doc_nr)
        if verbose:
            print("Tokenized", tokens)
        tokens_lm = self._lemmatize_tokens(tokens)
        if verbose:
            print("Lemmatized", tokens_lm)
        return tokens_lm

    def extract_common_words(self):
        filename = self.common_words_filename
        comp_text = " ".join(self._prepare_complete_text(doc) for doc in self.documents)
        term_array = self._prepare_text(comp_text, verbose=False)
        candidates = Counter(term_array).most_common(86)
        temp = []
        for k, v in candidates:
            if v >= self.freq_threshold:
                temp.append(k)

        with open(filename, "w", encoding="utf8") as f:
            f.write(str(temp))

    def parse_doc(self, docid, only_tokenize=False, remove_del=False, verbose=False):
        pass

    def get_docids(self):
        pass

    def parse_text(self, text, verbose=False, remove_del=False, only_tokenize=False):
        return self._prepare_text(text, verbose=verbose, remove_del=remove_del, only_tokenize=only_tokenize)

    def remove_commons_and_delimiters(self, elem):
        pass

    def _remove_commons_and_delimiters(self, elem, delimiters):
        res = ''
        for c in elem:
            if c not in delimiters:
                res += c
        if res not in self.common_words:
            return res
        return None


class EnglishParser(Parser):
    Delimiters = [';', '#', ')', '(', '.', ':', '/', '?', '\\', ',', '\n', '&']

    def __init__(self, preload_corpus=True, is_data_tagged=False, corpus_address="DataSet/corpus/English.csv"):
        common_words_filename = "DataSet/common_words/english_common_words"
        super().__init__(freq_threshold=100, common_words_filename=common_words_filename)

        nltk.download('wordnet')
        nltk.download('punkt')

        self.normalizer = str.lower
        self.lemmatizer = nltk.stem.WordNetLemmatizer().lemmatize
        self.word_tokenize = nltk.tokenize.word_tokenize
        self.is_data_tagged = is_data_tagged
        if is_data_tagged:
            self.first_row = []
        if preload_corpus:
            self.documents = self.read_english_documents(corpus_address)

    def _prepare_complete_text(self, doc):
        return doc

    def _is_delimiter(self, ch):
        return ch in EnglishParser.Delimiters

    def remove_commons_and_delimiters(self, elem):
        return self._remove_commons_and_delimiters(elem, EnglishParser.Delimiters)

    def read_english_documents(self, filename, set_docs=False):
        documents = []
        with open(filename, encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    if self.is_data_tagged:
                        documents.append(f'{row[1]} {row[2]}')
                        self.first_row.append(row[0])
                    else:
                        documents.append(f'{row[0]} {row[1]}')
                line_count += 1
            print(f'Processed {line_count} lines.')
        if set_docs:
            self.documents = documents
        return documents

    def load_english_documents(self, documents_data, is_data_tagged=True, has_header=True):
        documents = []
        if is_data_tagged:
            for i in range(1, len(documents_data)):
                documents.append(f'{documents_data[i][1]} {documents_data[i][2]}')
        else:
            for row in documents_data:
                documents.append(row)
        self.documents = documents

    def get_docids(self):
        return [i for i in range(len(self.documents))]

    def parse_doc(self, docid, only_tokenize=False, remove_del=False, verbose=False):
        return self._prepare_text(self.documents[docid], only_tokenize=only_tokenize, remove_del=remove_del,
                                  verbose=verbose)


class PersianParser(Parser):
    Delimiters = [
        ".", "[", "\n", "]", "{", "}", "\"", "'",
        "|", ",", " ", ":", "=", "(", ")",
        "*", "-", "/", "#", "<", ">", "~", "_", "،", "٫", "«", "»", "؟", "'", "۰"]

    def __init__(self):
        common_words_filename = "DataSet/common_words/persian_common_words"
        super().__init__(freq_threshold=2500, common_words_filename=common_words_filename)

        self.normalizer = Normalizer().normalize
        self.lemmatizer = Lemmatizer().lemmatize
        self.word_tokenize = word_tokenize

        handler = open("DataSet/corpus/Persian.xml", encoding="utf8").read()
        self.bs = BeautifulSoup(handler, features="lxml")
        self.pages = self.bs.find_all("page")
        self.documents = self.pages

    @staticmethod
    def _pick_desired_tags(page):
        text = page.text
        username = [""]
        comments = [""]
        if page.username:
            username = page.username.contents
        if page.comment:
            comments = page.comment.contents
        return text + " " + username[0] + " " + comments[0]

    def _is_delimiter(self, ch):
        return ch in PersianParser.Delimiters or Parser._is_english(ch)

    def _prepare_complete_text(self, doc):
        return self._pick_desired_tags(doc)

    def parse_doc(self, docid, only_tokenize=False, remove_del=False, verbose=False):
        return self.parse_page(docid, verbose=False, only_tokenize=False, remove_del=False)

    def parse_page(self, pageid, verbose=False, only_tokenize=False, remove_del=False):
        page = [p for p in self.pages if int(p.id.contents[0]) == pageid][0]
        doc = PersianParser._pick_desired_tags(page)
        return self._prepare_text(text=doc, verbose=verbose, only_tokenize=only_tokenize, remove_del=remove_del)

    def get_docids(self):
        return [int(p.id.contents[0]) for p in self.pages]

    def remove_commons_and_delimiters(self, elem):
        return self._remove_commons_and_delimiters(elem, PersianParser.Delimiters)


if __name__ == '__main__':
    p = PersianParser()
    # p = EnglishParser()
    # print(p.parse_page(p.get_docids()[-1]))
    # print(p.parse_text(input()))
    # for pp in p.get_docids():
    #     print(pp, end=" ")
    #     (p.parse_page(pp))
    p.extract_common_words()
    # p = EnglishParser("DataSet/English.csv")
    # p.extract_common_words("DataSet/common_words/english_common_words")
    # for id in p.get_docids():
    #     print(p.parse_doc(id))