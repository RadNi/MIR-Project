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

    def index(self):
        docids = self.parser.get_docids()
        posting_list = []
        for id in docids:
            ind = self.parser.parse_page(id)
            input(str(self.get_duplicates_with_info(ind)))
            self.merge_index(ind, posting_list)

    def index_doc(self, docId):
        tokens = self.parser.parse_page(docId)
        print(tokens)
        lemmatizer = Lemmatizer()
        for t in tokens:
            print(t, lemmatizer.lemmatize(t))

    def get_docids(self):
        return self.parser.get_docids()

    def merge_index(self, ind, posting_list):
        pass

    def get_duplicates_with_info(self, list_of_elems):
        ''' Get duplicate element in a list along with thier indices in list
         and frequency count'''
        dict_of_elems = dict()
        index = 0
        # Iterate over each element in list and keep track of index
        for elem in list_of_elems:
            print(elem)
            # If element exists in dict then keep its index in lisr & increment its frequency
            if not self.parser.must_delete(elem):
                # purified_elem = self.parser._prepare_text(elem)
                if elem in dict_of_elems:
                    dict_of_elems[elem].append(index)
                else:
                    # Add a new entry in dictionary
                    dict_of_elems[elem] = [index]
            index += 1

        dict_of_elems = {key: value for key, value in dict_of_elems.items()}
        return dict_of_elems


if __name__ == '__main__':
    ind = Indexer("DataSet/Persian.xml")
    ind.index()
    # print(ind.index_doc(ind.get_docids()[-1]))
