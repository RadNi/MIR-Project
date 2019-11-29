from Code.parser import *
from hazm import *


class Indexer:

    """
    شد شد#شو
    In cases like this what should we do? (Lemmatizing)
    """

    def __init__(self, filename):
        self.filename = filename
        self.parser = Parser(filename)
        self.persian_posting_list = dict()
        self.persian_bigram_index = {}

    def index_persian(self):
        docids = self.parser.get_docids()
        i = 1
        for id in docids:
            print(id, ":", i, "/", len(docids))
            i += 1
            ind = self.parser.parse_page(id)
            table = self.get_duplicates_with_info(ind)
            self.merge_index(table, id)
        self.write_index_to_file()

    # def index_doc(self, docId):
    #     tokens = self.parser.parse_page(docId)
    #     print(tokens)
    #     lemmatizer = Lemmatizer()
    #     for t in tokens:
    #         print(t, lemmatizer.lemmatize(t))

    # def get_docids(self):
    #     return self.parser.get_docids()

    def create_persian_bigram_index(self):

        docids = self.parser.get_docids()
        i = 1
        for id in docids:
            print(id, ":", i, "/", len(docids))
            i += 1
            all_terms = self._create_all_persian_terms(id)
            for term in all_terms:
                self._add_term_to_bigram(term)
        self._write_persian_bigram_to_file()

    def merge_index(self, table, id):
        for term in table:
            if term in self.persian_posting_list:
                self.persian_posting_list[term].append({id: table[term]})
            else:
                self.persian_posting_list[term] = [{id: table[term]}]

    def get_duplicates_with_info(self, list_of_elems):
        ''' Get duplicate element in a list along with their indices in list
         and frequency count '''
        dict_of_elems = dict()
        index = 0
        # Iterate over each element in list and keep track of index
        for elem in list_of_elems:
            # print(elem)
            # If element exists in dict then keep its index in lisr & increment its frequency
            new_elem = self.parser.remove_commons_and_delimiters(elem)
            if new_elem:
                prepared = self.parser.parse_text(new_elem, only_tokenize=False, remove_del=False)
                if len(prepared) > 0:
                    purified_elem = prepared[0]
                    if purified_elem in dict_of_elems:
                        dict_of_elems[purified_elem].append(index)
                    else:
                        # Add a new entry in dictionary
                        dict_of_elems[purified_elem] = [index]
            index += 1

        dict_of_elems = {key: value for key, value in dict_of_elems.items()}
        return dict_of_elems

    def write_index_to_file(self):
        with open("persian_index", "w") as f:
            f.write(str(self.persian_posting_list))

    def read_persian_index(self):
        with open("persian_index", "r") as f:
            return f.read()

    def _create_all_persian_terms(self, page_id):
        all_terms = set()
        ind = self.parser.parse_page(page_id, only_tokenize=True, remove_del=True, verbose=False)
        # input(ind)
        for term in ind:
            new_term = self.parser.remove_commons_and_delimiters(term)
            if new_term:
                all_terms.add(new_term)
        return all_terms

    def _add_term_to_bigram(self, term):
        l = list(term)
        for i in range(len(l) - 1):
            bigram = "".join(c for c in l[i:i + 2])
            if bigram in self.persian_bigram_index:
                if term not in self.persian_bigram_index[bigram]:
                    self.persian_bigram_index[bigram].append(term)
                # else:
                #     print(self.persian_bigram_index[bigram], bigram)
                #     input(term)
                # input()
            else:
                self.persian_bigram_index[bigram] = [term]

    def _write_persian_bigram_to_file(self):

        with open("persian_bigram", "w") as f:
            f.write(str(self.persian_bigram_index))


if __name__ == '__main__':
    ind = Indexer("DataSet/Persian.xml")
    ind.create_persian_bigram_index()
    # ind.index_persian()
    # ind.read_persian_index()
    # print(ind.index_doc(ind.get_docids()[-1]))
