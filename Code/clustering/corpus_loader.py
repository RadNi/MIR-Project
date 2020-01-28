import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from gensim.models import KeyedVectors
from sklearn.mixture import GaussianMixture

from Code.indexer import Indexer
from Code.printer import Printer
import _pickle
import numpy as np


class ClusterManager:
    def __init__(self, **kwargs):
        self.w2v_model = None
        self.w2v_vec_len = 0
        self.tf_idf_model = TfidfVectorizer(use_idf=True, smooth_idf=True, norm='l2')
        self.tf_idf_ready = False
        self.tf_idf_word_to_ind_map = {}
        self.id_list = []
        self.corpus_data = []
        self.corpus_w2v_vs = None
        self.corpus_w2v_valid_data = []
        self.corpus_tf_idf_vs = None
        self.corpus_tf_idf_vs_small = None
        self.indexer = Indexer("english", is_data_tagged=True, preload_corpus=False, **kwargs)

    def load_w2v_model(self, model_address, binary=True):
        Printer.print("Loading Word2Vec model: " + model_address)
        try:
            self.w2v_model = KeyedVectors.load_word2vec_format(model_address, binary=binary)
        except:
            with open(model_address, encoding="utf8") as f:
                self.w2v_model = {}
                line_num = 0
                Printer.indent_right()
                while True:
                    line_num += 1
                    line = f.readline()
                    if line is None or len(line) == 0:
                        break
                    split_line = line.split(" ")
                    word = split_line[0]
                    self.w2v_model[word] = np.array(list(map(float, split_line[1:])))
                    self.w2v_vec_len = len(split_line) - 1
                    if line_num % 10000 == 0:
                        Printer.print(f'Completed Line: {line_num}')
                Printer.indent_left()

        Printer.print("Finished loading Word2Vec model...")

    def load_corpus(self, corpus_address, corpus_encoding="latin1", remove_urls=True, remove_mentions=False):
        with open(corpus_address, encoding=corpus_encoding) as f:
            csv_reader = csv.reader(f, delimiter=',')
            head = True
            for line in csv_reader:
                if not head:
                    self.id_list.append(str(line[0]))
                    split_line = line[1].split(" ")
                    final_word_list = []
                    for word in split_line:
                        if remove_mentions and word.startswith("@"):
                            continue
                        if remove_urls and word.startswith("http://") or word.startswith("https://"):
                            continue
                        final_word_list.append(word)
                    removed_str = " ".join(final_word_list)
                    stemmed_list = self.indexer.parser.parse_text(removed_str, remove_del=True)
                    self.corpus_data.append(stemmed_list)
                head = False

    def corpus_generate_tf_idf_model(self):
        if len(self.corpus_data) == 0:
            Printer.print("Corpus not loaded!")
            return
        if not self.tf_idf_ready:
            Printer.print("Started creating corpus tf-idf model")
            corpus_joined_list = list(map(lambda x: " ".join(x), self.corpus_data))
            self.corpus_tf_idf_vs = self.tf_idf_model.fit_transform(corpus_joined_list)
            vec_sum = np.array(np.sum(self.corpus_tf_idf_vs, axis=0))[0]
            arg_sorted = np.argsort(vec_sum)
            slice_range = 12
            vec_indices = arg_sorted[vec_sum > vec_sum[arg_sorted[(len(arg_sorted) * (slice_range - 1)) // slice_range]]]
            self.corpus_tf_idf_vs_small = self.corpus_tf_idf_vs[:, vec_indices]
            feature_list = self.tf_idf_model.get_feature_names()
            for i, feature in enumerate(feature_list):
                self.tf_idf_word_to_ind_map[feature] = i
            self.tf_idf_ready = True
            Printer.print("Finished creating corpus tf-idf model")
        else:
            Printer.print("Corpus tf-idf model already loaded")

    def corpus_generate_word2vec_model(self, weight_with_tf_idf=True):
        if len(self.corpus_data) == 0:
            Printer.print("Corpus not loaded!")
            return
        if self.w2v_model is None:
            Printer.print("Word2Vec model not loaded!")
            return
        Printer.print("Started creating corpus Word2Vec model")
        if weight_with_tf_idf and not self.tf_idf_ready:
            self.corpus_generate_tf_idf_model()
        valid_data = []
        self.corpus_w2v_valid_data = []
        for i, word_list in enumerate(self.corpus_data):
            is_valid = False
            current_vec = np.zeros((self.w2v_vec_len,))
            if weight_with_tf_idf:
                for word in word_list:
                    try:
                        base_vec = self.w2v_model[word]
                        is_valid = True
                        current_vec += base_vec * self.corpus_tf_idf_vs[i, self.tf_idf_word_to_ind_map[word]]
                    except KeyError:
                        continue
            else:
                for word in word_list:
                    try:
                        current_vec += self.w2v_model[word]
                        is_valid = True
                    except KeyError:
                        continue
            self.corpus_w2v_valid_data.append(is_valid)
            if is_valid:
                valid_data.append(current_vec)
        self.corpus_w2v_vs = np.array(valid_data)
        Printer.print("Finished creating corpus Word2Vec model")

    def _w2v_result_write_to_file(self, output_file_address, labels):
        with open(output_file_address, mode="w") as f:
            i = 0
            for ind, check_valid in enumerate(self.corpus_w2v_valid_data):
                if check_valid:
                    f.write(f'{self.id_list[ind]}, {labels[i]}\n')
                    i += 1
                else:
                    f.write(f'{self.id_list[ind]}, N/A\n')

    def _tf_idf_result_write_to_file(self, output_file_address, labels):
        with open(output_file_address, mode="w") as f:
            for i, id_val in enumerate(self.id_list):
                f.write(f'{id_val}, {labels[i]}\n')

    def cluster_w2v_with_k_means(self, cluster_count, output_file_address=None):
        km = KMeans(n_clusters=cluster_count)
        distances = km.fit_transform(self.corpus_w2v_vs)
        inertia = self.get_inertia(distances, km.labels_)
        if output_file_address is not None:
            self._w2v_result_write_to_file(output_file_address, km.labels_)
        return inertia, km.labels_, distances

    def cluster_tf_idf_with_k_means(self, cluster_count, output_file_address=None):
        km = KMeans(n_clusters=cluster_count)
        distances = km.fit_transform(self.corpus_tf_idf_vs_small)
        inertia = self.get_inertia(distances, km.labels_)
        if output_file_address is not None:
            self._tf_idf_result_write_to_file(output_file_address, km.labels_)
        return inertia, km.labels_, distances

    def cluster_w2v_with_gaussian_mixture(self, cluster_count, output_file_address=None):
        gm = GaussianMixture(n_components=cluster_count, n_init=5)
        labels = gm.fit_predict(self.corpus_w2v_vs)
        score = gm.score(self.corpus_w2v_vs)
        if output_file_address is not None:
            self._w2v_result_write_to_file(output_file_address, labels)
        return score, labels

    def cluster_tf_idf_with_gaussian_mixture(self, cluster_count, output_file_address=None):
        gm = GaussianMixture(n_components=cluster_count, n_init=3)
        dense_corpus = self.corpus_tf_idf_vs_small.toarray()
        labels = gm.fit_predict(dense_corpus)
        score = gm.score(dense_corpus)
        if output_file_address is not None:
            self._tf_idf_result_write_to_file(output_file_address, labels)
        return score, labels

    def cluster_w2v_with_agglomerative_cluster(self, cluster_count, output_file_address=None):
        acm = AgglomerativeClustering(n_clusters=cluster_count)
        labels = acm.fit_predict(self.corpus_w2v_vs)
        if output_file_address is not None:
            self._w2v_result_write_to_file(output_file_address, labels)
        return labels, acm

    def cluster_tf_idf_with_agglomerative_cluster(self, cluster_count, output_file_address=None):
        acm = AgglomerativeClustering(n_clusters=cluster_count)
        labels = acm.fit_predict(self.corpus_tf_idf_vs_small.toarray())
        if output_file_address is not None:
            self._tf_idf_result_write_to_file(output_file_address, labels)
        return labels, acm

    @staticmethod
    def get_inertia(distances, labels):
        variance = 0
        i = 0
        for label in labels:
            variance = variance + distances[i][label]
            i = i + 1
        return variance
