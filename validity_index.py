import numpy as np
import pandas as pd
from validclust import dunn
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score as asi
from sklearn.metrics import davies_bouldin_score as dbi
from sklearn.metrics import calinski_harabasz_score as chi

from kmeans import kmeans as kms
from kmeans import three_way_kmeans as twkms
from kmeans import get_cores_fringes
from kmeans import get_data_labels


def calculate_dbi(data: np.ndarray, clusters: list[np.ndarray]) -> float:
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


def calculate_silhouette_score(data: np.ndarray, clusters: list[np.ndarray]) -> float:
    """
    TODO 此函数的计算结果和sklearn.metrics.silhouette_score函数的结果一致！！！
    计算所有样本的平均轮廓系数Average Silhouette index, [-1, 1], 平均轮廓系数越接近1越好
    """
    k = len(clusters)
    # TODO 让每一个簇的核心域参与计算，不要包括边缘域
    # 如果clusters是K-Means产生的，那么cores和clusters相同(因为边界域是空集)
    _, cores, _ = get_cores_fringes(clusters)

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
                distance = pairwise_distances(data[cores[i]], data[cores[j]])
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


def calculate_chi(data: np.ndarray, clusters: list[np.ndarray]) -> float:
    """
    TODO 此函数的计算结果和sklearn.metrics.calinski_harabasz_score函数的结果一致！！！
    计算calinski harabasz score(方差比准则, 簇间紧密度与簇内紧密度的比值)
    """
    k = len(clusters)

    # TODO 让每一个簇的核心域参与计算，不要包括边缘域
    # 如果clusters是K-Means产生的，那么cores和clusters相同(因为边界域是空集)
    _, cores, _ = get_cores_fringes(clusters)
    # cores_1d是cores的一维形式
    cores_1d = np.unique(np.concatenate(cores))

    # 计算核心域中的样本数量n和样本的特征数m
    n = len(cores_1d)
    m = data.shape[1]

    # 计算核心域的全局中心
    global_centroid = np.mean(data[cores_1d], axis=0)

    # 计算每个簇的中心
    centroids = list()
    for i in range(k):
        centroids.append(np.mean(data[cores[i]], axis=0))
    centroids = np.array(centroids)

    # 计算类间散布矩阵b_matrices
    b_matrices = np.full((m, m), 0, dtype=np.float64)
    for i in range(k):
        centroid = centroids[i]
        n_i = len(cores[i])
        matrix = n_i * np.outer(centroid-global_centroid, centroid-global_centroid)
        b_matrices += matrix

    # 计算类内散布矩阵w_matrices
    w_matrices = np.full((m, m), 0, dtype=np.float64)
    for i in range(k):
        centroid = centroids[i]
        matrix = np.full((m, m), 0, dtype=np.float64)
        for sample_idx in cores[i]:
            matrix += np.outer(data[sample_idx]-centroid, data[sample_idx]-centroid)
        w_matrices += matrix

    # 计算方差比准则
    return (np.trace(b_matrices) / (k-1)) / (np.trace(w_matrices) / (n-k))


def calculate_dunn(data: np.ndarray, clusters: list[np.ndarray]) -> float:
    """
    TODO 此函数的计算结果和validclust.dunn函数的结果一致！！！
    dunn指数的值越大表示聚类效果越好; dunn越大, 簇间距离大 (簇分离得好) 且簇内距离小 (簇紧密)
    """
    k = len(clusters)

    # TODO 让每一个簇的核心域参与计算，不要包括边缘域
    # 如果clusters是K-Means产生的，那么cores和clusters相同(因为边界域是空集)
    _, cores, _ = get_cores_fringes(clusters)

    # 计算每一对簇之间的簇间距离(用簇i和簇j最近样本间的距离表示)
    intra_dists = list()
    for i in range(k):
        for j in range(i, k):
            if j != i:
                intra_dists.append(np.min(pairwise_distances(data[cores[i]], data[cores[j]])))

    # 计算每一个簇的簇内距离
    inter_dists = list()
    for i in range(k):
        inter_dists.append(np.max(pairwise_distances(data[cores[i]])))

    # 计算dunn值
    return np.min(intra_dists) / np.max(inter_dists)


def main() -> None:
    """计算K-Means和3WK-Means算法的DBI"""
    # 导入数据
    data = np.loadtxt('sample.txt')

    # 对于K-Means, get_labels_data返回的数据集就是data的副本
    _, clusters = kms(data, k=3)
    _, labels = get_data_labels(data, clusters)

    # 获取核心域的数据集和每一个样本的簇标签
    _, clusters_3w = twkms(data, k=3, epsilon=2.64)
    cores_data, cores_labels = get_data_labels(data, clusters_3w)

    # 使用自己写的函数计算K-Means的validity indices
    dbi_kms = calculate_dbi(data, clusters)
    asi_kms = calculate_silhouette_score(data, clusters)
    chi_kms = calculate_chi(data, clusters)
    dunn_kms = calculate_dunn(data, clusters)

    # 使用第三方库函数计算K-Means的validity indices
    dbi_kms_std = dbi(data, labels)
    asi_kms_std = asi(data, labels)
    chi_kms_std = chi(data, labels)
    dunn_kms_std = dunn(pairwise_distances(data), labels)

    # 使用自己写的函数计算3WK-Means的validity indices
    dbi_3w = calculate_dbi(data, clusters_3w)
    asi_3w = calculate_silhouette_score(data, clusters_3w)
    chi_3w = calculate_chi(data, clusters_3w)
    dunn_3w = calculate_dunn(data, clusters_3w)

    # 使用第三方库函数计算3WK-Means的validity indices
    dbi_3w_std = dbi(cores_data, cores_labels)
    asi_3w_std = asi(cores_data, cores_labels)
    chi_3w_std = chi(cores_data, cores_labels)
    dunn_3w_std = dunn(pairwise_distances(cores_data), cores_labels)

    # 设置字典和DataFrame
    index = ['DBI', 'ASI', 'CHI', 'DUNN']
    dictionary = {
        "K-Means(std)": np.round([dbi_kms_std, asi_kms_std, chi_kms_std, dunn_kms_std], 4),
        "K-Means": np.round([dbi_kms, asi_kms, chi_kms, dunn_kms], 4),
        "3WK-Means": np.round([dbi_3w, asi_3w, chi_3w, dunn_3w], 4),
        "3WK-Means(std)": np.round([dbi_3w_std, asi_3w_std, chi_3w_std, dunn_3w_std], 4),
    }
    df = pd.DataFrame(dictionary, index=index)
    print(df)


if __name__ == '__main__':
    main()
