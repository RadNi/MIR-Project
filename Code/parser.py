from bs4 import BeautifulSoup
from hazm import *
from collections import Counter

"""
    Currently we only consider persian texts
"""


class Parser:

    """
    Shall we ignore nim-fasele?
    What's the policy?
    """

    Delimiters = [
        ".", "[", "\n", "]", "{", "}", "\"", "'",
        "|", ",", " ", ":", "=", "(", ")",
        "*", "-", "/", "#", "<", ">", "~", "_", "،", "٫", "«", "»", "؟", "'"]

    @staticmethod
    def _is_english(ch):
        return 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' or '0' <= ch <= '9'

    @staticmethod
    def _is_delimiter(ch):
        return ch in Parser.Delimiters or Parser._is_english(ch)

    def __init__(self, filename):
        handler = open(filename).read()
        self.bs = BeautifulSoup(handler, features="lxml")
        self.pages = self.bs.find_all("page")
        self.normalizer = Normalizer()
        self.lemmatizer = Lemmatizer().lemmatize
        self.freq_threshould = 40
        self.common_words = self._read_common_words()

    def extract_common_words(self):
        comp_text = " ".join(self._pick_desired_tags(p) for p in self.pages)
        term_array = self._prepare_text(comp_text, verbose=False)
        candidates = Counter(term_array).most_common(86)
        temp = []
        for k, v in candidates:
            if v >= 2500:
                temp.append(k)

        with open("common_terms", "w") as f:
            f.write(str(temp))

    def parse(self):
        self._find_highfreq_terms()

    def _find_highfreq_terms(self):
        for p in self.pages:
            input(self._pick_desired_tags(p))
        # sum_str = " ".join()

    # def must_delete(self, term):
    #     return self._is_delimiter(term) or (term in self.common_words)

    def _rm_delimiters(self, document):
        res = list(document)
        for i in range(len(document)):
            if Parser._is_delimiter(document[i]):
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
        doc = Parser._pick_desired_tags(page)
        return self._prepare_text(text=doc, verbose=verbose, only_tokenize=only_tokenize, remove_del=remove_del)

    def get_docids(self):
        return [int(p.id.contents[0]) for p in self.pages]

    def parse_text(self, text, verbose=False, remove_del=False, only_tokenize=False):
        return self._prepare_text(text, verbose=verbose, remove_del=remove_del, only_tokenize=only_tokenize)

    def _rm_highfreq_tokens(self, tokens):
        print(list(filter(lambda x: Counter(tokens)[x] >= sum(Counter(tokens).values()) / self.freq_threshould,
                     Counter(tokens).keys())))

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
    def remove_commons_and_delimiters(self, elem):
        res = ''
        for c in elem:
            if c not in Parser.Delimiters:
                res += c
        if res not in self.common_words:
            return res
        return None


if __name__ == '__main__':
    p = Parser("DataSet/Persian.xml")
    # print(p.parse_page(p.get_docids()[-1]))
    # print(p.parse_text(input()))
    # for pp in p.get_docids():
    #     print(pp, end=" ")
    #     (p.parse_page(pp))
    # print(p.extract_common_words())