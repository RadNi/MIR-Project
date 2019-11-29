from Code.query_processor import QueryCorrector
from Code.parser import Parser
from Code.utils import VectorSpace
import numpy as np
import scipy.sparse


class Searcher:
    # FIXME should get from somewhere
    TOTAL_DOCUMENTS = 1000

    def __init__(self, query_corrector: QueryCorrector, postings_index: dict, parser,
                 vector_space: VectorSpace):
        self.postings_index = postings_index
        self.query_corrector = query_corrector
        self.parser = parser
        self.word2index = list(postings_index.keys())
        self.vector_space = vector_space

    def search(self, query, mode):
        corrected_query = self.query_corrector.correct_query(query, mode)
        print(corrected_query)
        normalized_query = self.parser.parse_text(text=corrected_query)
        current_posting: dict = self.postings_index.get(normalized_query[0])
        # Intersect all doc IDs
        doc_id_set = set(current_posting.keys())
        for word in set(normalized_query):
            current_posting_list: dict = self.postings_index.get(word)
            doc_id_set.intersection_update(set(current_posting_list.keys()))
        # rank by tf-idf
        query_vector = self.vector_space.calculate_query_vec(normalized_query)
        doc_id_list = list(doc_id_set)
        doc_id_list.sort(key=lambda doc_id: np.dot(self.vector_space.get_doc_vec(doc_id).T, query_vector)[0, 0], reverse=True)
        if len(doc_id_list) > 15:
            doc_id_list = doc_id_list[:15]
        return doc_id_list

    def proximity_search(self, query, window_size:int, mode):
        corrected_query = self.query_corrector.correct_query(query, mode)
        normalized_query = self.parser.parse_text(text=corrected_query)
        current_posting_list: dict = self.postings_index.get(normalized_query[0])
        # Intersect all doc IDs
        doc_id_set = set(current_posting_list.keys())
        for word in set(normalized_query):
            current_posting_list: dict = self.postings_index.get(word)
            doc_id_set.intersection_update(set(current_posting_list.keys()))
        # Check proximity
        final_doc_list = []
        for doc in doc_id_set:
            smallest_last_index = max(self.postings_index.get(normalized_query[0]).get(doc))
            # finding smallest last word index
            for word in set(normalized_query):
                other_last_index = max(self.postings_index.get(word).get(doc))
                if other_last_index < smallest_last_index:
                    smallest_last_index = smallest_last_index
            # moving window checking
            found_window = False
            for window_start in range(smallest_last_index + 1):
                window_okay = True
                for word in set(normalized_query):
                    position_array = np.array(self.postings_index.get(word).get(doc))
                    if not np.any(window_start <= position_array < (window_start + window_size)):
                        window_okay = False
                        break
                if window_okay:
                    found_window = True
                    break
            if found_window:
                final_doc_list.append(doc)

        # rank by tf-idf
        query_vector = self.vector_space.calculate_query_vec(normalized_query)
        final_doc_list.sort(key=lambda doc_id: scipy.sparse.lil_matrix.dot(self.vector_space.get_doc_vec(doc_id).T, query_vector)[0, 0], reverse=True)
        if len(final_doc_list) > 15:
            final_doc_list = final_doc_list[:15]
        return final_doc_list



