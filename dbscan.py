import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances


def range_query(dataset: np.ndarray, i: int, epsilon: float) -> np.ndarray:
    """求样本点p的epsilon邻域"""
    distances = pairwise_distances(dataset, dataset[[i]])
    judge_distances = distances <= epsilon
    return np.where(judge_distances.flatten() == 1)[0]


def dbscan(dataset: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """动手实现DBSCAN算法(参考英文维基百科)"""
    n_samples = len(dataset)

    # 标签为-2，表示此样本点未被访问过
    labels = np.full(n_samples, -2)
    c = 0
    for i in range(n_samples):
        if labels[i] != -2:
            continue  # 如果点已经被处理，跳过

        # 获取样本点i的邻域
        neighbors = range_query(dataset, i, eps)

        if neighbors.size < min_samples:
            labels[i] = -1  # 噪声点的标签为-1
            continue

        # 如果样本点i既没有被处理过，也不是噪声点
        labels[i] = c

        # 获取样本点i的种子集合(不包含i)
        seed_set = np.delete(neighbors, np.where(neighbors == i)[0])
        seed_set = set(seed_set)

        # 处理样本点i邻域中所有的点
        while seed_set:
            j = seed_set.pop()
            if labels[j] == -1:
                labels[j] = c
            if labels[j] != -2:
                continue

            labels[j] = c
            j_neighbors = range_query(dataset, j, eps)
            if j_neighbors.size >= min_samples:
                seed_set.update(set(j_neighbors))

        c += 1

    return labels


def visualize_dbscan_result(dataset: np.ndarray, labels: np.ndarray) -> None:
    """可视化sklean中的DBSCAN算法"""
    print(np.unique(labels))
    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels + 1, cmap='viridis', s=15, marker='o')
    plt.show()


def main():
    dataset = np.loadtxt('sample.txt')

    # 使用sklearn中的DBSCAN算法
    # noinspection PyUnresolvedReferences
    labels_std = DBSCAN(eps=0.5, min_samples=5).fit(dataset).labels_

    # 使用自己实现的DBSCAN算法
    labels = dbscan(dataset, eps=0.5, min_samples=5)

    # 可视化聚类结果
    visualize_dbscan_result(dataset, labels_std)
    visualize_dbscan_result(dataset, labels)


if __name__ == '__main__':
    main()
