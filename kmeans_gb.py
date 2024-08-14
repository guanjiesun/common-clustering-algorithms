import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances

from gbs import GranularBall
from gbs import generate_gbs
from gbs import verify_gbs
from gbs import visualize_gbs
from visualize_original_data import visualize_original_data


def gb_kmeans(gb_centroids: np.ndarray, k: int = 3, max_iterations: int = 100,
              tolerance: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """
    TODO 对粒球空间进行KMeans聚类
    gb_centroids表示每一个粒球的质心组成的二维numpy数组，对这n_gb个质心进行KMeans聚类即可
    """

    # n_gb表示粒球空间中粒球的数量
    n_gb = len(gb_centroids)

    # 判断data, k, max_iterations是否合法
    if not isinstance(gb_centroids, np.ndarray) or gb_centroids.ndim != 2:
        raise TypeError("Data must be a 2D numpy array!")
    if k <= 0 or k > n_gb or max_iterations < 1:
        raise ValueError("Invalid number of clusters!")

    # 随机初始化k个聚类中心和样本的簇标签
    np.random.seed(100)
    centers = gb_centroids[np.random.choice(n_gb, k, replace=False)]

    # 开始迭代
    n_iteration = 0
    while True:
        n_iteration += 1
        # distances, shape=(k, n_gb); 表示k个聚类中心到n_samples个数据点的距离
        distances = pairwise_distances(centers, gb_centroids)

        # 基于距离矩阵，计算每一个样本的簇标签
        gb_labels = np.argmin(distances, axis=0)

        # 计算新的聚类中心
        new_centers = np.array([np.mean(gb_centroids[gb_labels == i], axis=0) for i in range(k)])

        # 检查算法是否收敛
        if np.all(np.abs(new_centers - centers) < tolerance) or n_iteration > max_iterations:
            # 新聚类中心和旧聚类中心相比变化很小或者达到最大迭代次数, 则停止迭代
            break

        # 若算法未收敛, 则更新聚类中心, 进行下一次迭代
        centers = new_centers

    # centers, shape=(k, m_features), 表示k个聚类中心; gb_labels表示每一个粒球的簇标签
    return centers, gb_labels


def get_sample_labels(gbs: list[GranularBall], gb_labels: np.ndarray) -> np.ndarray:
    """获取每一个样本的簇标签"""

    # 先计算原始数据集中的样本数量
    n_samples = 0
    for gb in gbs:
        n_samples += len(gb.indices)

    # 初始化样本簇标签
    labels = np.full(n_samples, -1)

    for i, gb in enumerate(gbs):
        for sample_idx in gb.indices:
            # 粒球包含的样本的簇标签和粒球的簇标签保持一直
            labels[sample_idx] = gb_labels[i]

    return labels


def visualize_gb_kmeans_result(dataset: np.ndarray, sample_labels: np.ndarray) -> None:
    fig, ax = plt.subplots()
    ax.scatter(dataset[:, 0], dataset[:, 1], c=sample_labels, cmap='viridis', s=5, marker='.')
    ax.set_title('GB-KMeans Clustering')
    plt.show()


def main():
    """GB-KMeans算法实现"""
    # 载入数据
    dataset = np.loadtxt('sample.txt')

    # 生成粒球空间
    gbs = generate_gbs(dataset)

    # 验证粒球空间的有效性
    verify_gbs(gbs)

    # 可视化原始数据
    visualize_original_data(dataset)

    # 可视化粒球空间
    visualize_gbs(gbs)

    # 对粒球空间进行KMeans聚类
    centers, gb_labels = gb_kmeans(np.array([gb.centroid for gb in gbs]), k=3)

    # 获取每一个样本的簇标签
    sample_labels = get_sample_labels(gbs, gb_labels)

    # 可视化GB-KMeans聚类结果
    visualize_gb_kmeans_result(dataset, sample_labels)


if __name__ == '__main__':
    main()
