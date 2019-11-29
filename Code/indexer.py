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
        self.posting_list = dict()

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

    def merge_index(self, table, id):
        for term in table:
            if term in self.posting_list:
                self.posting_list[term].append({id: table[term]})
            else:
                self.posting_list[term] = [{id: table[term]}]

    def get_duplicates_with_info(self, list_of_elems):
        ''' Get duplicate element in a list along with their indices in list
         and frequency count '''
        dict_of_elems = dict()
        index = 0
        # Iterate over each element in list and keep track of index
        for elem in list_of_elems:
            # print(elem)
            # If element exists in dict then keep its index in lisr & increment its frequency
            if not self.parser.must_delete(elem):
                prepared = self.parser.parse_text(elem)
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
            f.write(str(self.posting_list))

    def read_persian_index(self):
        with open("persian_index", "r") as f:
            return f.read()


if __name__ == '__main__':
    ind = Indexer("DataSet/Persian.xml")
    # ind.index_persian()
    ind.read_persian_index()
    # print(ind.index_doc(ind.get_docids()[-1]))
