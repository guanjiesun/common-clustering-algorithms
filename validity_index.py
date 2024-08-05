import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

from kmeans import kmeans as kms
from kmeans import three_way_kmeans as twkms
from kmeans import get_cores_fringes


def calculate_dbi(data: np.ndarray, clusters: list[np.ndarray]):
    """
    TODO 此函数的计算结果和sklearn.metrics.davies_bouldin_score函数的结果一致！！！
    DBI是聚类算法的一个validity index(performance metrics), DBI越小越好
    DBI定义参考王平心三支K-Means的论文，不要参考周志华《机器学习》上关于DBI的定义
    """
    k = len(clusters)
    # TODO 让每一个簇的核心域参与计算，不要包括边缘域
    # 如果clusters是K-Means产生的，那么cores和clusters相同
    _, cores, _ = get_cores_fringes(clusters)

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


def calculate_silhouette_score(data: np.ndarray, clusters: list[np.ndarray]):
    """
    TODO 此函数的计算结果和sklearn.metrics.davies_bouldin_score函数的结果一致！！！
    计算所有样本的平均轮廓系数Average Silhouette index, [-1, 1], 平均轮廓系数越接近1越好
    """
    k = len(clusters)
    # TODO 让每一个簇的核心域参与计算，不要包括边缘域
    # 如果clusters是K-Means产生的，那么cores和clusters相同(因为边界域是空集)
    _, cores, _ = get_cores_fringes(clusters)
    # cores_1d = np.unique(np.concatenate(cores))  # cores_1d是cores的一维形式

    # 计算每一个点到簇内其他点距离的平均值
    distances1 = list()
    for i in range(k):
        # 簇i内数据点的距离矩阵
        distances = pairwise_distances(data[cores[i]])
        distances1.append(np.sum(distances, axis=1) / (len(cores[i])-1))

    # 计算簇每一个点到其他簇的平均距离中的最小值
    distances2 = list()
    for i in range(k):
        distances = list()
        for j in range(k):
            if j != i:
                # distance存储簇i的所有点到簇j的所有点的距离
                distance = cdist(data[cores[i]], data[cores[j]])
                distances.append(np.sum(distance, axis=1) / len(cores[j]))

        distances = np.array(distances)
        distances2.append(np.min(distances, axis=0))

    # 计算每一个点的轮廓
    silhouette = list()
    for i in range(k):
        for j in range(len(cores[i])):
            b = distances2[i][j]
            a = distances1[i][j]
            s = (b-a) / np.max([a, b])
            silhouette.append(s)

    # 计算数据集的平均轮廓
    return np.mean(silhouette)


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
    # 导入数据
    data = np.loadtxt('sample.txt')

    # 应用聚类算法
    _, clusters = kms(data, k=3)
    labels = get_labels(clusters)
    _, clusters_3w = twkms(data, k=3, epsilon=2.64)

    # 计算DBI
    dbi_score = calculate_dbi(data, clusters)
    dbi_score_3w = calculate_dbi(data, clusters_3w)
    dbi_score_std = davies_bouldin_score(data, labels)

    # 计算ASI
    asi_score = calculate_silhouette_score(data, clusters)
    asi_score_3w = calculate_silhouette_score(data, clusters_3w)
    asi_score_std = silhouette_score(data, labels)

    # 打印结果
    print(np.round(dbi_score, 4))
    print(np.round(dbi_score_std, 4))
    print(np.round(dbi_score_3w, 4))
    print(np.round(asi_score_std, 4))
    print(np.round(asi_score, 4))
    print(np.round(asi_score_3w, 4))


if __name__ == '__main__':
    main()
