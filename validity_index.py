import numpy as np
import sklearn.cluster as sc
from sklearn.metrics import davies_bouldin_score as dbi

from kmeans import kmeans
from kmeans import three_way_kmeans
from kmeans import get_cores_fringes


def get_dbi(data: np.ndarray, clusters: list):
    # 初始化所有样本的簇标签为1
    labels = [-1 for _ in range(len(data))]
    k = len(clusters)
    for i in range(k):
        for sample_index in clusters[i]:
            labels[sample_index] = i

    labels = np.array(labels)
    dbi_score = dbi(data, labels)
    return np.round(dbi_score, 4)


def my_dbi():
    pass


def main() -> None:
    """计算K-Means和3WK-Means算法的DBI"""
    data = np.loadtxt('sample.txt')
    _, clusters1 = kmeans(data, k=3)
    _, clusters2 = three_way_kmeans(data, k=3, epsilon=2.64)
    clusters2, cores2, fringes2 = get_cores_fringes(clusters2)

    dbi_score_1 = get_dbi(data, clusters1)
    dbi_score_2 = get_dbi(data, clusters2)
    print(dbi_score_1, dbi_score_2)

    labels = sc.KMeans(n_clusters=3).fit_predict(data)
    dbi_score_3 = dbi(data, labels)
    print(np.round(dbi_score_3, 4))


if __name__ == '__main__':
    main()
