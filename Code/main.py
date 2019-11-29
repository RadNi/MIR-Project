from Code.query_processor import QueryCorrector
from Code.searcher import Searcher
from Code.parser import PersianParser, EnglishParser
from Code.utils import VectorSpace
from Code.indexer import Indexer


def print_line():
    print("-------------------------------\n")


def invalid():
    print("Invalid input! Input the number of selection")
    print_line()


if __name__ == '__main__':
    mode = "persian"
    while True:
        mode = input("""
        Select language:
        1. English
        2. Farsi
        """)
        print_line()
        if mode == 1:
            mode = "english"
            break
        elif mode == 2:
            mode = "persian"
            break
        else:
            invalid()
    # FIXME stub
    query_corrector = QueryCorrector(list())
    searcher = Searcher(query_corrector, {}, None, None)
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
        if selection == 0:
            exit(0)
        elif selection == 1:
            # Pre-processing scenarios
            pass

        elif selection == 2:
            # Indexing scenarios
            pass

        elif selection == 3:
            # Query correction scenarios
            while True:
                selection2 = input("""
                Select what you want to do:
                \t1. Correct a query
                
                \t0. Back
                """)
                if selection2 == 0:
                    print_line()
                    break
                elif selection2 == 1:
                    # Get a query and fix
                    query = input(f"""
                    Enter your query, language is {mode}:
                    """)
                    print(query_corrector.correct_query(query, mode))
                    print_line()
                else:
                    invalid()

        elif selection == 4:
            # Search scenarios
            while True:
                selection2 = input("""
                Select what you want to do:
                \t1. Normal Search
                \t2. Proximity Search
                
                \t0. Back
                """)
                if selection2 == 0:
                    print_line()
                    break
                elif selection2 == 1:
                    query = input(f"""
                    Enter your query, language is {mode}:
                    """)
                    print(f"Output is:\n {searcher.search(query)}")
                    print_line()
                elif selection2 == 2:
                    query = input(f"""
                    Enter your query, language is {mode}:
                    """)
                    proximity_range = 0
                    while True:
                        proximity_range_str = input("""
                        Enter proximity window size:
                        """)
                        try:
                            proximity_range = int(proximity_range_str)
                            if proximity_range <= 0:
                                print("Enter an integer bigger than 0:")
                                continue
                            print(f"Output is: \n{searcher.proximity_search(query, proximity_range)}")
                            print_line()
                            break
                        except ValueError:
                            print("Enter an integer bigger than 0:")
                            continue

                else:
                    invalid()

        else:
            invalid()
