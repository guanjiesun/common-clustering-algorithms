import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from visualize_original_data import visualize_original_data


def kmeans(data: np.ndarray, k: int = 3, max_iterations: int = 100,
           tolerance: float = 1e-4) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    K-Means算法实现(将kmeans升级为kmeans++)
    :param data: np.ndarray, shape=(n_samples, n_features), ndim=2
    :param k: number of clusters
    :param max_iterations: the number of maximum iterations
    :param tolerance: float, tolerance for convergence
    :return: tuple[np.ndarray, np.ndarray]
        - centroids: np.ndarray, shape=(k, n_features), cluster centers
        - labels: np.ndarray, shape=(n_samples), the cluster label for each sample
    """
    n_samples = len(data)

    # 判断data, k, max_iterations是否合法
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError("Data must be a 2D numpy array!")
    if k <= 0 or k > n_samples or max_iterations < 1:
        raise ValueError("Invalid number of clusters!")

    # 初始化k个聚类中心和样本的簇标签
    np.random.seed(100)
    centroids = data[np.random.choice(n_samples, k, replace=False)]

    # 开始迭代
    n_iteration = 0
    while True:
        n_iteration += 1
        # distances, shape=(k, n_samples); 表示k个聚类中心到n_samples个数据点的距离
        distances = pairwise_distances(centroids, data)

        # 基于距离矩阵，计算每一个样本的簇标签
        labels = np.argmin(distances, axis=0)

        # 基于簇标签，保存每一个簇中的每一个样本在数据集data中的索引
        clusters = list()
        for i in range(k):
            # 每一个cluster存储簇i的样本在数据集data中的索引
            cluster = np.where(labels == i)[0]
            clusters.append(cluster)

        # 计算新的聚类中心
        new_centroids = np.array([np.mean(data[clusters[i]], axis=0) for i in range(k)])

        # 检查算法是否收敛
        if np.all(np.abs(new_centroids - centroids) < tolerance) or n_iteration > max_iterations:
            # 新聚类中心和旧聚类中心相比变化很小或者达到最大迭代次数, 则停止迭代
            break

        # 若算法未收敛, 则更新聚类中心, 进行下一次迭代
        centroids = new_centroids

    # centroids, shape=(k, m_features), 表示k个聚类中心; clusters: list[np.ndarray], len(clusters)=k
    return centroids, clusters


def visualize_kmeans_result(data: np.ndarray, centroids: np.ndarray,
                            clusters: list[np.ndarray]) -> None:
    """
    可视化K-Means聚类结果
    :param data: np.ndarray, shape=(n_samples, m_features), ndim=2
    :param centroids: np.ndarray, shape=(k, m_features), ndim=1
    :param clusters: np.ndarray, shape=(n_samples), the cluster label for each sample
    :return: None
    """
    fig, ax = plt.subplots()
    # 簇的个数k和数据集中的样本数量n_samples
    k = len(clusters)
    n_samples = len(data)

    # 为每一个样本分配簇标签
    labels = np.full(n_samples, -1)
    for i in range(k):
        for sample_idx in clusters[i]:
            labels[sample_idx] = i
    labels = np.array(labels)

    # 绘制数据点
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='plasma', s=5, marker='.')
    # 绘制聚类中心
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, marker='x')
    # 设置图片属性和样式
    ax.set_title('K-Means Clustering')
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def main() -> None:
    """
    1. 数据集sample.txt, (4000, 2)
    2. K-Means, k=3
    3. 可视化KMeans聚类结果
    :return: None
    """

    # 载入数据
    dataset = np.loadtxt('sample.txt')

    # 获取KMeans聚类结果
    centroids, clusters = kmeans(dataset, k=3)

    # 可视化原始数据
    visualize_original_data(dataset)

    # 可视化KMeans聚类结果
    visualize_kmeans_result(dataset, centroids, clusters)


if __name__ == '__main__':
    main()
