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
                else:
                    invalid()

        elif selection == 4:
            # Search scenarios
            pass

        else:
            invalid()




