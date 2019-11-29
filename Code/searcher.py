from Code.query_processor import QueryCorrector
from Code.parser import Parser
from Code.utils import VectorSpace
import numpy as np


class Searcher:
    # FIXME should get from somewhere
    TOTAL_DOCUMENTS = 1000

    def __init__(self, query_corrector: QueryCorrector, postings_index: dict, parser: Parser,
                 vector_space: VectorSpace):
        self.postings_index = postings_index
        self.query_corrector = query_corrector
        self.parser = parser
        self.word2index = list(postings_index.keys())
        self.vector_space = vector_space

    def search(self, query):
        corrected_query = self.query_corrector.correct_query(query)
        normalized_query = self.parser.parse_text(corrected_query)
        current_posting_list = self.postings_index.get(normalized_query[0])
        # Intersect all doc IDs
        doc_id_set = set([posting.keys() for posting in current_posting_list])
        for word in set(normalized_query):
            current_posting_list = self.postings_index.get(word)
            doc_id_set.intersection_update(set([posting.keys() for posting in current_posting_list]))
        # rank by tf-idf
        query_vector = self.vector_space.calculate_query_vec(normalized_query)
        doc_id_list = list(doc_id_set)
        doc_id_list.sort(key=lambda doc_id: np.dot(self.vector_space.doc_dict[doc_id], query_vector), reverse=True)
        if len(doc_id_list) > 15:
            doc_id_list = doc_id_list[:15]
        return doc_id_list
