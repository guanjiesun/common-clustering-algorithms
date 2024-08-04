import numpy as np
from sklearn.metrics import pairwise_distances
# from sklearn.metrics import silhouette_score
# from sklearn.metrics import davies_bouldin_score

from kmeans import kmeans as kms
from kmeans import three_way_kmeans as twkms


def calculate_dbi(data: np.ndarray, clusters: list[np.ndarray]):
    """
    DBI是聚类算法的一个validity index(performance metrics)
    定义参考周志华《机器学习》
    """
    k = len(clusters)

    # 计算每一个簇的中心; centroids, list[np.ndarray]; np.ndarray.shape=(k, m)
    centroids = list()
    for i in range(k):
        # centroid是簇i的中心
        centroid = np.mean(data[clusters[i]], axis=0)
        centroids.append(centroid)

    # 计算簇中心之间的距离矩阵; c_distances, np.ndarray, shape=(k, k)
    c_distances = pairwise_distances(centroids)

    # 计算每一个簇包含的样本数量
    numbers = list()
    for i in range(k):
        # number是簇i包含的样本数量
        number = len(clusters[i])
        numbers.append(number)

    # 计算每一个簇样本之间的距离的矩阵; distances, np.ndarray, shape=(n, n)
    distances = list()
    for i in range(k):
        # distance是簇i的距离矩阵, 函数默认使用的距离度量是欧式距离
        distance = pairwise_distances(data[clusters[i]])
        distances.append(distance)

    # 计算每一个簇的样本间的平均距离; averages, list[float]
    averages = list()
    for i in range(k):
        distance = distances[i]
        total_distance = np.sum(distance) / 2
        samples_number = numbers[i]
        count = (samples_number * (samples_number-1)) / 2
        average_distance = total_distance / count
        averages.append(average_distance)

    # 计算每一个簇和其他簇之间的相似度比率
    similarity_ratios_list = list()
    for i in range(k):
        # similarity_ratios表示簇i的相似度比率列表
        similarity_ratios = list()
        for j in range(k):
            if i != j:
                similarity_ratio = (averages[i]+averages[j]) / (c_distances[i, j])
                similarity_ratios.append(similarity_ratio)
        similarity_ratios_list.append(similarity_ratios)

    # 计算每一个簇的dbi(簇和其他簇之间的最大相似度比率)
    dbi_list = list()
    for i in range(k):
        # dbi_score表示簇i的dbi值
        dbi_score = np.max(similarity_ratios_list[i])
        dbi_list.append(dbi_score)

    a = np.mean(dbi_list)
    # 计算聚类算法的dbi
    return np.mean(dbi_list)


def main() -> None:
    """计算K-Means和3WK-Means算法的DBI"""
    data = np.loadtxt('sample.txt')
    _, clusters = kms(data, k=3)
    _, clusters1 = twkms(data, k=3, epsilon=2.64)
    my_score = calculate_dbi(data, clusters)
    my_score1 = calculate_dbi(data, clusters1)
    print(np.round(my_score, 4), np.round(my_score1, 4))


if __name__ == '__main__':
    main()
