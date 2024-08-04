import numpy as np
from sklearn.metrics import silhouette_score as ss
from sklearn.metrics import davies_bouldin_score as dbi

from kmeans import kmeans as kms
from kmeans import three_way_kmeans as twkms
from kmeans import get_cores_labels_data


def main() -> None:
    """计算K-Means和3WK-Means算法的DBI"""
    data = np.loadtxt('sample.txt')
    # import pandas as pd
    # data = pd.read_csv('wine.csv').to_numpy()

    _, labels = kms(data, k=3)
    _, clusters1 = twkms(data, k=3, epsilon=2.64)
    labels1, cores_data1 = get_cores_labels_data(data, clusters1)

    dbi_score = dbi(data, labels)
    dbi_score_1 = dbi(cores_data1, labels1)
    ss1 = ss(data, labels)
    ss2 = ss(cores_data1, labels1)
    print(np.round(dbi_score, 4), np.round(dbi_score_1, 4))
    print(np.round(ss1, 4), np.round(ss2, 4))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(cores_data1[:, 0], cores_data1[:, 1], c=labels1, cmap='viridis', s=10, marker='.')
    plt.show()


if __name__ == '__main__':
    main()
