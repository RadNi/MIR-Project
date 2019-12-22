from Code.parser import *
from Code.constants import *


class Indexer:

    def __init__(self, mode, preload_corpus=True, is_data_tagged=False,
                 bigram_index_file_name="DataSet/bigram_tables/english_bigram",
                 index_filename="DataSet/indexes/english_index"):
        if mode == 'persian':
            self.parser = PersianParser()
            self.bigram_index_filename = "DataSet/bigram_tables/persian_bigram"
            self.index_filename = "DataSet/indexes/persian_index"
        elif mode == 'english':
            self.parser = EnglishParser(is_data_tagged=is_data_tagged, preload_corpus=preload_corpus)
            self.bigram_index_filename = bigram_index_file_name
            self.index_filename = index_filename
        self.posting_list = dict()
        self.bigram_index = {}

    def set_adresses(self, index_address, bigram_addres):
        self.index_filename = index_address
        self.bigram_index_filename = bigram_addres

    def _merge_index(self, table, id):
        for term in table:
            if term in self.posting_list:
                self.posting_list[term][id] = table[term]
            else:
                self.posting_list[term] = {id: table[term]}

    def _get_duplicates_with_info(self, list_of_elems):
        """
        Get duplicate element in a list along with their indices in list
        and frequency count
        """
        dict_of_elems = dict()
        index = 0
        # Iterate over each element in list and keep track of index
        for elem in list_of_elems:
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

    def _write_index_to_file(self, file_name=None):
        if file_name is None:
            with open(self.index_filename, "w", encoding="utf8") as f:
                f.write(str(self.posting_list))
        else:
            with open(file_name, "w", encoding="utf8") as f:
                f.write(str(self.posting_list))

    def create_all_terms(self, page_id):
        all_terms = set()
        ind = self.parser.parse_doc(page_id, only_tokenize=True, remove_del=True, verbose=False)
        for term in ind:
            new_term = self.parser.remove_commons_and_delimiters(term)
            if new_term:
                all_terms.add(new_term)
        return all_terms

    def _add_term_to_bigram(self, term):
        l = list(term)
        for i in range(len(l) - 1):
            bigram = "".join(c for c in l[i:i + 2])
            if bigram in self.bigram_index:
                if term not in self.bigram_index[bigram]:
                    self.bigram_index[bigram].append(term)
            else:
                self.bigram_index[bigram] = [term]

    def _write_bigram_to_file(self):
        with open(self.bigram_index_filename, "w", encoding="utf8") as f:
            f.write(str(self.bigram_index))

    def read_index_table(self, index_address=None):
        print("Reading index table ...")
        if index_address is None:
            with open(self.index_filename, "r", encoding="utf8") as f:
                return eval(f.read())
        else:
            with open(index_address, "r", encoding="utf8") as f:
                return eval(f.read())

    def read_bigram(self):
        with open(self.bigram_index_filename, "r", encoding="utf8") as f:
            return eval(f.read())

    def index_single_doc(self, docid):
        index = self.parser.parse_doc(docid)
        return self._get_duplicates_with_info(index)

    def create_bigram_index(self):
        docids = self.parser.get_docids()
        i = 1
        for id in docids:
            print(id, ":", i, "/", len(docids))
            i += 1
            all_terms = self.create_all_terms(id)
            for term in all_terms:
                self._add_term_to_bigram(term)
        self._write_bigram_to_file()

    def index(self, should_write=True, file_name=None):
        docids = self.parser.get_docids()
        i = 1
        for id in docids:
            print(id, ":", i, "/", len(docids))
            i += 1
            ind = self.parser.parse_doc(id)
            table = self._get_duplicates_with_info(ind)
            self._merge_index(table, id)
        if should_write:
            self._write_index_to_file(file_name)


if __name__ == '__main__':
    ind = Indexer(MODE)
    ind.index()
    ind.create_bigram_index()
    # ind.index_persian()
    # ind.read_persian_bigram()
    # print(ind.index_doc(ind.get_docids()[-1]))
