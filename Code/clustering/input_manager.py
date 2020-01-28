from Code.printer import Printer
from Code.clustering.corpus_loader import ClusterManager
from Code.main import print_line
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def invalid():
    Printer.print("Invalid input!")
    print_line()


# Ref: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    distance = np.arange(model.children_.shape[0])

    linkage_matrix = np.column_stack([model.children_, distance,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def k_means_control():
    print_line()
    Printer.indent_right()
    while True:
        Printer.print("Select Vector Space Type:")
        Printer.indent_right()
        Printer.print("""1. Word2Vec Space
        2. Tf-Idf Space
        
        0. Back""")
        Printer.indent_left()
        inp1 = input()
        if inp1 == '1':
            print_line()
            Printer.indent_right()
            Printer.print("Word2Vec Space Selected")
            output_list = []
            while True:
                print_line()
                Printer.print("Enter Cluster Count: (Enter 0 for ending)")
                try:
                    count = int(input())
                except:
                    invalid()
                    continue
                if count <= 0:
                    Printer.indent_left()
                    break
                inertia, labels, distances = cm.cluster_w2v_with_k_means(count)
                Printer.print(f'Clustering Finished, Inertia: {inertia}')
                output_list.append((count, inertia))
                output_list.sort(key=lambda x: x[0])
                labels = []
                values = []
                for output in output_list:
                    labels.append(output[0])
                    values.append(output[1])
                plt.title('K-Means Word2Vec Inertia Plot')
                plt.plot(labels, values, linewidth=2, color='blue', marker='o')
                plt.show()
        elif inp1 == '2':
            print_line()
            Printer.indent_right()
            Printer.print("Tf-Idf Space Selected")
            output_list = []
            while True:
                print_line()
                Printer.print("Enter Cluster Count: (Enter 0 for ending)")
                try:
                    count = int(input())
                except:
                    invalid()
                    continue
                if count <= 0:
                    Printer.indent_left()
                    break
                inertia, labels, distances = cm.cluster_tf_idf_with_k_means(count)
                Printer.print(f'Clustering Finished, Inertia: {inertia}')
                output_list.append((count, inertia))
                output_list.sort(key=lambda x: x[0])
                labels = []
                values = []
                for output in output_list:
                    labels.append(output[0])
                    values.append(output[1])
                plt.title('K-Means Tf-Idf Inertia Plot')
                plt.plot(labels, values, linewidth=2, color='blue', marker='o')
                plt.show()
        elif inp1 == '0':
            Printer.indent_left()
            return
        else:
            invalid()


def gaussian_control():
    print_line()
    Printer.indent_right()
    while True:
        Printer.print("Select Vector Space Type:")
        Printer.indent_right()
        Printer.print("""1. Word2Vec Space
            2. Tf-Idf Space

            0. Back""")
        Printer.indent_left()
        inp1 = input()
        if inp1 == '1':
            print_line()
            Printer.indent_right()
            Printer.print("Word2Vec Space Selected")
            output_list = []
            while True:
                print_line()
                Printer.print("Enter Cluster Count: (Enter 0 for ending)")
                try:
                    count = int(input())
                except:
                    invalid()
                    continue
                if count <= 0:
                    Printer.indent_left()
                    break
                score, labels = cm.cluster_w2v_with_gaussian_mixture(count)
                total_score = np.sum(score)
                Printer.print(f'Clustering Finished, Score: {total_score}')
                output_list.append((count, total_score))
                output_list.sort(key=lambda x: x[0])
                labels = []
                values = []
                for output in output_list:
                    labels.append(output[0])
                    values.append(output[1])
                plt.title('Gaussian Mixture Word2Vec Score Plot')
                plt.plot(labels, values, linewidth=2, color='red', marker='o')
                plt.show()
        elif inp1 == '2':
            print_line()
            Printer.indent_right()
            Printer.print("Tf-Idf Space Selected")
            output_list = []
            while True:
                print_line()
                Printer.print("Enter Cluster Count: (Enter 0 for ending)")
                try:
                    count = int(input())
                except:
                    invalid()
                    continue
                if count <= 0:
                    Printer.indent_left()
                    break
                score, labels = cm.cluster_tf_idf_with_gaussian_mixture(count)
                total_score = np.sum(score)
                Printer.print(f'Clustering Finished, Score: {total_score}')
                output_list.append((count, total_score))
                output_list.sort(key=lambda x: x[0])
                labels = []
                values = []
                for output in output_list:
                    labels.append(output[0])
                    values.append(output[1])
                plt.title('Gaussian Mixture Tf-Idf Score Plot')
                plt.plot(labels, values, linewidth=2, color='red', marker='o')
                plt.show()
        elif inp1 == '0':
            Printer.indent_left()
            return
        else:
            invalid()


def hierarchical_control():
    print_line()
    Printer.indent_right()
    while True:
        Printer.print("Select Vector Space Type:")
        Printer.indent_right()
        Printer.print("""1. Word2Vec Space
                2. Tf-Idf Space

                0. Back""")
        Printer.indent_left()
        inp1 = input()
        if inp1 == '1':
            print_line()
            Printer.indent_right()
            Printer.print("Word2Vec Space Selected")
            output_list = []
            while True:
                print_line()
                Printer.print("Enter Cluster Count: (Enter 0 for ending)")
                try:
                    count = int(input())
                except:
                    invalid()
                    continue
                if count <= 0:
                    Printer.indent_left()
                    break
                labels, model = cm.cluster_w2v_with_agglomerative_cluster(count)
                Printer.print(f'Clustering Finished')
                plt.title('Hierarchical Clustering Dendrogram')
                plot_dendrogram(model, truncate_mode='lastp', p=count)
                plt.show()

        elif inp1 == '2':
            print_line()
            Printer.indent_right()
            Printer.print("Tf-Idf Space Selected")
            output_list = []
            while True:
                print_line()
                Printer.print("Enter Cluster Count: (Enter 0 for ending)")
                try:
                    count = int(input())
                except:
                    invalid()
                    continue
                if count <= 0:
                    Printer.indent_left()
                    break
                labels, model = cm.cluster_tf_idf_with_agglomerative_cluster(count)
                Printer.print(f'Clustering Finished')
                plt.title('Hierarchical Clustering Dendrogram')
                plot_dendrogram(model, truncate_mode='lastp', p=count)
                plt.show()
        elif inp1 == '0':
            Printer.indent_left()
            return
        else:
            invalid()


if __name__ == '__main__':
    cm = ClusterManager()
    cm.load_corpus('DataSet/corpus/Phase3_Data.csv', remove_mentions=True)
    cm.load_w2v_model('DataSet/models/deps.words', binary=False)
    cm.corpus_generate_tf_idf_model()
    cm.corpus_generate_word2vec_model()
    print_line()
    while True:
        Printer.print("Select Clustering Method:")
        Printer.indent_right()
        Printer.print("""1. K-Means
        2. Gaussian Mixture
        3. Hierarchical Clustering
        
        0. Exit""")
        Printer.indent_left()
        selection = input()
        if selection == '1':
            k_means_control()
        elif selection == '2':
            gaussian_control()
        elif selection == '3':
            hierarchical_control()
        elif selection == '0':
            exit(0)
        else:
            invalid()
