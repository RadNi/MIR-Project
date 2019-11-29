from bs4 import BeautifulSoup
from hazm import *
from collections import Counter


class Parser:

    """
    Shall we ignore nim-fasele?
    What's the policy?
    """

    @staticmethod
    def _is_english(ch):
        return 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' or '0' <= ch <= '9'

    def __init__(self):
        self.normalizer = Normalizer()
        self.lemmatizer = Lemmatizer().lemmatize
        # self.freq_threshould = 40
        self.common_words = self._read_common_words()
        self.documents = []

    # def _prepare_complete_text(self, doc):
    #     pass

    def extract_common_words(self):
        # for doc in self.documents:
        #     print(self._prepare_complete_text(doc))
        comp_text = " ".join(self._prepare_complete_text(doc) for doc in self.documents)
        # print(comp_text)
        term_array = self._prepare_text(comp_text, verbose=False)
        candidates = Counter(term_array).most_common(86)
        temp = []
        for k, v in candidates:
            if v >= 2500:
                temp.append(k)

        with open("common_terms_2", "w") as f:
            f.write(str(temp))

    # def _find_highfreq_terms(self):
    #     for p in self.pages:
    #         input(self._pick_desired_tags(p))
        # sum_str = " ".join()

    # def must_delete(self, term):
    #     return self._is_delimiter(term) or (term in self.common_words)

    def _rm_delimiters(self, document):
        res = list(document)
        for i in range(len(document)):
            if self._is_delimiter(document[i]):
                res[i] = " "
        return "".join(c for c in res)

    def _is_delimiter(self, doc):
        pass

    def _normalize_doc(self, doc):
        pass

    def _tokenize(self, doc):
        pass

    def _lemmatize_tokens(self, tokens):
        pass

    def _prepare_text(self, text, remove_del=False, verbose=False, only_tokenize=False):
        pass

    def get_docids(self):
        pass

    def parse_text(self, text, verbose=False, remove_del=False, only_tokenize=False):
        return self._prepare_text(text, verbose=verbose, remove_del=remove_del, only_tokenize=only_tokenize)
    #
    # def _rm_highfreq_tokens(self, tokens):
    #     print(list(filter(lambda x: Counter(tokens)[x] >= sum(Counter(tokens).values()) / self.freq_threshould,
    #                  Counter(tokens).keys())))

    def _read_common_words(self):
        with open("common_terms", 'r') as f:
            return eval(f.read())

    # def purify(self, elem):
    #     return elem
    #     new_elem = ''
    #     for c in elem:
    #         if not self._is_delimiter(c):
    #             new_elem += c
    #     return new_elem

    def _remove_commons_and_delimiters(self, elem, delimiters):
        res = ''
        for c in elem:
            if c not in delimiters:
                res += c
        if res not in self.common_words:
            return res
        return None

    def remove_commons_and_delimiters(self, elem):
        pass


class EnglishParser(Parser):
    pass


class PersianParser(Parser):

    Delimiters = [
        ".", "[", "\n", "]", "{", "}", "\"", "'",
        "|", ",", " ", ":", "=", "(", ")",
        "*", "-", "/", "#", "<", ">", "~", "_", "،", "٫", "«", "»", "؟", "'", "۰"]

    def _prepare_complete_text(self, doc):
        return self._pick_desired_tags(doc)

    def remove_commons_and_delimiters(self, elem):
        return self._remove_commons_and_delimiters(elem, PersianParser.Delimiters)

    def __init__(self, filename):
        super().__init__()
        handler = open(filename).read()
        self.bs = BeautifulSoup(handler, features="lxml")
        self.pages = self.bs.find_all("page")
        self.documents = self.pages

    def _is_delimiter(self, ch):
        return ch in PersianParser.Delimiters or Parser._is_english(ch)

    def _rm_delimiters(self, document):
        res = list(document)
        for i in range(len(document)):
            if self._is_delimiter(document[i]):
                res[i] = " "
        return "".join(c for c in res)

    def _normalize_doc(self, doc):
        return self.normalizer.normalize("".join(doc))

    def _tokenize(self, doc):
        return word_tokenize(doc)

    def _lemmatize_tokens(self, tokens):
        res = []
        for t in tokens:
            res.append(self.lemmatizer(t))

        return res

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
        # final_tokens = self._rm_highfreq_tokens(tokens_lm)
        return tokens_lm

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

    def parse_page(self, pageid, verbose=False, only_tokenize=False, remove_del=False):
        page = [p for p in self.pages if int(p.id.contents[0]) == pageid][0]
        doc = PersianParser._pick_desired_tags(page)
        return self._prepare_text(text=doc, verbose=verbose, only_tokenize=only_tokenize, remove_del=remove_del)

    def get_docids(self):
        return [int(p.id.contents[0]) for p in self.pages]


if __name__ == '__main__':
    p = PersianParser("DataSet/Persian.xml")
    # print(p.parse_page(p.get_docids()[-1]))
    # print(p.parse_text(input()))
    # for pp in p.get_docids():
    #     print(pp, end=" ")
    #     (p.parse_page(pp))
    p.extract_common_words()
