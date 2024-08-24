import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances


def visualize_dbscan_result(dataset: np.ndarray, labels: np.ndarray) -> None:
    """可视化DBSCAN聚类结果"""
    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='plasma', s=5, marker='o')
    plt.title("DBSCAN")
    plt.show()


def visualize_dbscan_only_core_points(dataset: np.ndarray, labels: np.ndarray, core_indices: np.ndarray) -> None:
    """可视化DBSCAN聚类结果(只绘制核心点，不绘制边界点和噪声点)"""
    core_data = dataset[core_indices]
    core_labels = labels[core_indices]
    plt.scatter(core_data[:, 0], core_data[:, 1], c=core_labels, cmap='plasma', s=5, marker='o')
    plt.title("DBSCAN (Only Core Points)")
    plt.show()


def range_query(data: np.ndarray, idx: int, eps: float) -> np.ndarray:
    """获取样本点i的epsilon邻域"""
    distances = pairwise_distances(data, data[[idx]])
    judge_distances = (distances <= eps)
    return np.where(judge_distances.flatten() == 1)[0]


def get_three_way_dbscan_result(dataset: np.ndarray, labels: np.ndarray,
                                core_indices: np.ndarray, noise_indices: np.ndarray,
                                border_indices: np.ndarray, eps) -> list:
    """
    每一个簇由正域和边界域表示
    返回值：result=[[POS1, BND1], [POS2, BND2], [POS3, BND3],...,[POSk, BNDk]]
    """

    # 先计算簇的个数k
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

    # 再将噪声点分配到相应簇的边缘域
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


def visualize_three_way_dbscan_result(dataset: np.ndarray, labels: np.ndarray, clusters: list) -> None:
    # 先计算簇的个数
    if -1 in labels:
        k = len(np.unique(labels)) - 1
    else:
        k = len(np.unique(labels))

    # 计算所有簇的边界域的并集
    bnd_indices = set()
    for i in range(k):
        cluster = clusters[i]
        bnd_indices.update(cluster[1])

    # 绘制数据集边缘域的点
    bnd_data = dataset[list(bnd_indices)]
    plt.scatter(bnd_data[:, 0], bnd_data[:, 1], c='black', s=5, marker='o')

    # 绘制每一个簇的核心域
    colors = [
        'green', 'red', 'blue', 'purple', 'yellow', 'orange',
        'pink', 'brown', 'black', 'white', 'gray', 'cyan',
        'magenta', 'teal', 'navy', 'maroon', 'olive', 'lime',
        'indigo', 'violet', 'turquoise', 'gold', 'silver', 'beige',
        'lavender', 'coral', 'crimson', 'salmon', 'khaki',
        'plum', 'orchid', 'azure', 'charcoal', 'ivory',
        'mint', 'periwinkle', 'mauve', 'tan', 'burgundy'
    ]
    assert k <= len(colors)
    for i in range(k):
        cluster = clusters[i]
        pos_indices = cluster[0]
        pos_data = dataset[list(pos_indices)]
        plt.scatter(pos_data[:, 0], pos_data[:, 1], c=colors[i], s=5, marker='o')

    plt.title("3W-DBSCAN")
    plt.show()


def main():
    """基于DBSCAN的结果，进行三支决策"""
    dataset = np.loadtxt('sample.txt')  # 0.3, 5
    eps, min_samples = 0.3, 5

    # 获取DBSCAN聚类结果
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dataset)
    labels = clustering.labels_

    # 可视化DBSCAN聚类结果
    visualize_dbscan_result(dataset, labels)
    print(np.unique(labels))

    # 获取核心点、噪声点和边界点
    all_pts = np.array(range(len(dataset)))
    core_pts = clustering.core_sample_indices_
    noise_pts = np.where(labels == -1)[0]
    border_pts = np.array(list(set(all_pts) - set(core_pts) - set(noise_pts)))

    # 获取3W-DBSCAN聚类结果
    clusters = get_three_way_dbscan_result(dataset, labels, core_pts, noise_pts, border_pts, eps)

    # 可视化3W-DBSCAN聚类结果
    visualize_three_way_dbscan_result(dataset, labels, clusters)


if __name__ == '__main__':
    main()
