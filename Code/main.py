from Code.query_processor import QueryCorrector
from Code.searcher import Searcher
from Code.parser import PersianParser, EnglishParser
from Code.utils import VectorSpace
from Code.indexer import Indexer
from Code.classifiers.random_forest import RandomForestClassifier
from Code.classifiers.svm import SVMClassifier
from Code.classifiers.knn import KNNClassifier
from Code.classifiers.naive_bayes import NaiveBayesClassifier

def print_line():
    print("-------------------------------\n")


def invalid():
    print("Invalid input! Input the number of selection")
    print_line()


def pre_process_text(mode):

    if mode == 'persian':
        p = PersianParser()
    elif mode == 'english':
        p = EnglishParser()

    selection = input("""
Which part do you want to test?
\t1. Parse
\t2. Extract common words
""")

    if selection == "1":
        doc_id = input("""
Choose your document id to parse
""")
        print(p.parse_doc(doc_id))
    elif selection == "2":
        p.extract_common_words()


def train_classifier(classifier):
    train_set = classifier.read_data_from_file("DataSet/phase2/phase2_train.csv")
    classifier.train(train_set)
    # classifier.test(classifier.read_data_from_file("DataSet/phase2/phase2_test.csv"))


def choose_class():
    cls = int(input("""
Choose your desired class.
\t1. 
\t2. 
\t3. 
\t4. 
"""))
    return cls


def classifier_search():
    clsf_search = input("""
Choose your classifier.
\t1. KNN
\t2. SVM
\t3. Random Forest
\t4. Naive Bayes
""")
    if clsf_search == "3":
        return RandomForestClassifier()
    elif clsf_search == "2":
        penalty = float(input("Get penalty"))
        return SVMClassifier(penalty)
    elif clsf_search == "1":
        return KNNClassifier()
    elif clsf_search == "4":
        return NaiveBayesClassifier()


def create_index_tables(mode):
    indexer = Indexer(mode)

    selection = input("""
Which table do you want to create?
\t1. Index table
\t2. bi-gram table
""")

    if selection == "1":
        indexer.index()
    elif selection == "2":
        indexer.create_bigram_index()


def create_search_basedon_class(classifier, label):
    indexer = classifier.customised_index_creator(label)
    index_table = indexer.read_index_table()
    bigram_table = indexer.read_bigram()
    vector_space_model = classifier.read_vector_space_model(label)
    vector_space = VectorSpace(index_table, model=vector_space_model)
    query_corrector = QueryCorrector(bigram_table)
    return Searcher(query_corrector, index_table, parser, vector_space)


# def show_documents(searcher, docids):
#     query = input(f"""
#         Do you want to see actual documents?
#         """)
#     if query:
#         for docid in docids:
#             searcher.parser.read_english_documents(searcher.parser.doc_address)
#             print(searcher.parser.documents)
#             print(searcher.parser.documents[docid])
#     print_line()

def do_normal_and_proximity_search(selection2, searcher):
    if selection2 == "0":
        print_line()
        return False
    elif selection2 == "1":
        query = input(f"""
    Enter your query, language is {mode}:
    """)
        docids = searcher.search(query, mode)
        print(f"Output is:\n {docids}")
        # show_documents(searcher, docids)

        print_line()
    elif selection2 == "2":
        query = input(f"""
    Enter your query, language is {mode}:
    """)
        proximity_range = 0
        while True:
            proximity_range_str = input("""
    Enter proximity window size:
    """)
            proximity_range = int(proximity_range_str)
            if proximity_range <= 0:
                print("Enter an integer bigger than 0:")
                continue
            docids = searcher.proximity_search(query, proximity_range, mode)
            print(f"Output is: \n{docids}")
            print_line()
            # show_documents(searcher, docids)
            return False
    return True


if __name__ == '__main__':
    mode = "persian"
    while True:
        mode = input("""
Select language:
1. English
2. Farsi
""")
        print_line()
        if mode == "1":
            mode = "english"
            break
        elif mode == "2":
            mode = "persian"
            break
        else:
            invalid()
    if mode == "english":
        parser = EnglishParser()
    else:
        parser = PersianParser()
    indexer = Indexer(mode)
    index_table = indexer.read_index_table()
    bigram_table = indexer.read_bigram()
    vector_space_model = VectorSpace.read_vector_space_model(mode)
    vector_space = VectorSpace(index_table, model=vector_space_model)
    query_corrector = QueryCorrector(bigram_table)
    searcher = Searcher(query_corrector, index_table, parser, vector_space)
    while True:
        selection = input("""
Which part do you want to test?
\t1. Pre-processing (Parsing)
\t2. Indexing
\t3. Query correction
\t4. Search

\t0. Exit
""")
        print_line()
        if selection == "0":
            exit(0)
        elif selection == "1":
            # Pre-processing scenarios

            pre_process_text(mode)

        elif selection == "2":
            create_index_tables(mode)

        elif selection == "3":
            # Query correction scenarios
            while True:
                selection2 = input("""
Select what you want to do:
\t1. Correct a query

\t0. Back
""")
                if selection2 == "0":
                    print_line()
                    break
                elif selection2 == "1":
                    # Get a query and fix
                    query = input(f"""
Enter your query, language is {mode}:
""")
                    print(f"Output is: \n{query_corrector.correct_query(query, mode)}")
                    print_line()
                else:
                    invalid()

        elif selection == "4":
            # Search scenarios
            while True:
                selection2 = input("""
Select what you want to do:
\t1. Normal Search
\t2. Proximity Search
\t3. Classifier

\t0. Back
""")
                if selection2 == "1" or selection2 == "2":
                    flag = do_normal_and_proximity_search(selection2, searcher)
                    if not flag:
                        break
                elif selection2 == "3":
                    classifier = classifier_search()
                    cls = choose_class()
                    searcherr = create_search_basedon_class(classifier, cls)

                    selection3 = input("""
                    Select what you want to do:
                    \t1. Normal Search
                    \t2. Proximity Search

                    \t0. Back
                    """)
                    do_normal_and_proximity_search(selection3, searcherr)
                else:
                    invalid()

        else:
            invalid()
