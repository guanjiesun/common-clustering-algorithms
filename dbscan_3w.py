import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances


def visualize_dbscan_result(dataset: np.ndarray, labels: np.ndarray) -> None:
    """可视化sklean中的DBSCAN算法"""
    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='plasma', s=15, marker='o')
    plt.title("DBSCAN Clustering")
    plt.show()


def visualize_dbscan_only_core_points(dataset: np.ndarray, labels: np.ndarray, core_indices: np.ndarray) -> None:
    """只绘制DBSCAN聚类的核心点"""
    core_data = dataset[core_indices]
    core_labels = labels[core_indices]
    plt.scatter(core_data[:, 0], core_data[:, 1], c=core_labels, cmap='plasma', s=15, marker='o')
    plt.title("DBSCAN Clustering(Only Core Points)")
    plt.show()


def range_query(data: np.ndarray, idx: int, eps: float) -> np.ndarray:
    """获取样本点i的epsilon邻域"""
    distances = pairwise_distances(data, data[[idx]])
    judge_distances = (distances <= eps)
    return np.where(judge_distances.flatten() == 1)[0]


def get_three_way_dbscan_result(dataset: np.ndarray, labels: np.ndarray,
                                core_indices: np.ndarray, noise_indices: np.ndarray,
                                border_indices: np.ndarray, eps):
    """返回值是一个列表，每一个列表元素都是包含两个集合的列表"""
    # 先计算簇的个数
    if -1 in labels:
        k = len(np.unique(labels)) - 1
    else:
        k = len(np.unique(labels))

    clusters = list()
    for i in range(k):
        cluster = [set(), set()]
        clusters.append(cluster)

    # 先将核心点分配给相应簇的正域
    for core_idx in core_indices:
        clusters[labels[core_idx]][0].add(core_idx)

    # 再将噪声点分配到相应簇的边界域
    core_data = dataset[core_indices]
    for noise_idx in noise_indices:
        # 找出距离噪声点最近的核心点
        noise = dataset[noise_idx]
        distances = pairwise_distances(core_data, noise.reshape(1, -1))
        core_idx = np.argmin(distances)  # core_idx就是距离noise_idx最近的核心点
        # 将噪声点分配到core_idx所在簇的边缘域
        clusters[labels[core_idx]][1].add(noise_idx)

    # 最后将边界点分配到相应簇的正域或者某几个簇的边缘域
    for border_idx in border_indices:
        neighbors = range_query(dataset, border_idx, eps)
        neighbors_labels = np.unique(labels[neighbors])
        if neighbors_labels.size == 1:
            # 如果边界点所有邻居都属于同一个簇，则将此边界点加入到此簇的核心域
            clusters[neighbors_labels[0]][0].add(border_idx)
        else:
            # 如果边界点的邻居属于若干个簇，那么将边界点加入到这些簇的边界域
            for neighbor_label in neighbors_labels:
                clusters[neighbor_label][1].add(border_idx)

    return clusters


def main():
    """基于DBSCAN的结果，进一步分配"""
    dataset = np.loadtxt('sample.txt')  # 0.5, 5
    eps, min_samples = 0.5, 5

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dataset)
    # noinspection PyUnresolvedReferences
    labels = clustering.labels_
    visualize_dbscan_result(dataset, labels)
    print(np.unique(labels))

    all_indices = np.array(range(len(dataset)))
    # noinspection PyUnresolvedReferences
    core_indices = clustering.core_sample_indices_
    noise_indices = np.where(labels == -1)[0]
    border_indices = np.array(list(set(all_indices) - set(core_indices) - set(noise_indices)))

    visualize_dbscan_only_core_points(dataset, labels, core_indices)

    # 获取三支聚类的结果
    clusters = get_three_way_dbscan_result(dataset, labels, core_indices, noise_indices, border_indices, eps)
    print(len(clusters))


if __name__ == '__main__':
    main()
