from Code.query_processor import QueryCorrector

class Searcher:

    def __init__(self, query_corrector: QueryCorrector, postings_index, normalizer):
        # TODO create search tree
        self.postings_index = postings_index
        self.query_corrector = query_corrector
        # TODO initialize normalizer
        self.normalizer = normalizer

