import numpy as np
from sklearn.metrics import pairwise_distances
# from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

from kmeans import kmeans as kms
from kmeans import three_way_kmeans as twkms
from kmeans import get_cores_fringes


def calculate_dbi(data: np.ndarray, clusters: list[np.ndarray]):
    """
    TODO 此计算结果和sklearn中的结果一致！！！
    DBI是聚类算法的一个validity index(performance metrics)
    DBI定义参考王平心三支K-Means的论文，不要参考周志华《机器学习》上关于DBI的定义
    """
    k = len(clusters)
    # TODO 让每一个簇的核心域参与计算，不要包括边缘域
    _, cores, _ = get_cores_fringes(clusters)
    cores = [list(core) for core in cores]

    # 计算每一个簇的中心; centroids, list[np.ndarray]; np.ndarray.shape=(k, m)
    centroids = list()
    for i in range(k):
        # centroid是簇i的中心
        centroid = np.mean(data[cores[i]], axis=0)
        centroids.append(centroid)

    # 计算簇中心之间的距离矩阵; c_distances, shape=(k, k)
    c_distances = pairwise_distances(centroids)

    # 计算每一个簇的簇内距离intra cluster distance (icd hereafter)
    icd_list = list()
    for i in range(k):
        icd = np.sum(np.sqrt(np.sum(np.square(data[cores[i]] - centroids[i]), axis=1))) / len(cores[i])
        icd_list.append(icd)

    # 计算每一个簇和其它簇之间的相似度比率
    similarity_ratios_list = list()
    for i in range(k):
        # similarity_ratios表示簇i的和其它所有簇的相似度比率列表
        similarity_ratios = list()
        for j in range(k):
            if i != j:
                # similarity_ratio表示簇i和簇j的相似度比率
                similarity_ratio = (icd_list[i]+icd_list[j]) / (c_distances[i, j])
                similarity_ratios.append(similarity_ratio)
        similarity_ratios_list.append(similarity_ratios)

    # 计算每一个簇的dbi(簇和其他簇之间的最大相似度比率)
    dbi_list = list()
    for i in range(k):
        # dbi_score表示簇i的dbi值
        dbi_score = np.max(similarity_ratios_list[i])
        dbi_list.append(dbi_score)

    # 计算聚类算法的dbi
    return np.mean(dbi_list)


def get_labels(clusters: list[np.ndarray]) -> np.ndarray:
    """根据K-Means生成的clusters，求出每一个样本的簇标签"""
    k = len(clusters)

    # 计算样本总数
    n_samples = 0
    for i in range(k):
        n_samples += len(clusters[i])

    # 初始化每一个样本的簇标签
    labels = np.full(n_samples, -1)
    # 给每一个样本赋值一个簇标签
    for i in range(k):
        for sample_idx in clusters[i]:
            labels[sample_idx] = i

    return labels


def main() -> None:
    """计算K-Means和3WK-Means算法的DBI"""
    data = np.loadtxt('sample.txt')
    _, clusters = kms(data, k=3)
    _, clusters1 = twkms(data, k=3, epsilon=2.64)
    my_score = calculate_dbi(data, clusters)
    my_score1 = calculate_dbi(data, clusters1)
    labels = get_labels(clusters)
    std_score = davies_bouldin_score(data, labels)
    print(my_score, my_score1, std_score)


if __name__ == '__main__':
    main()
