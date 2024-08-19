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

        # 如果样本点i既没有被处理过，也不是噪声点，那么i就是核心点，给i分配一个簇标签
        labels[i] = c

        # 获取样本点i的种子集合(不包含i)
        seed_set = np.delete(neighbors, np.where(neighbors == i)[0])
        seed_set = set(seed_set)

        # 处理样本点i邻域中所有的点
        while seed_set:
            j = seed_set.pop()
            if labels[j] == -1:
                labels[j] = c  # 如果i的邻域j之前的标签是-1，那么让j的标签和i保持一致，即让j变成边界点
            if labels[j] != -2:
                continue  # 如果j已经被处理过，则跳过循环，继续处理样本点i的其他邻域点

            labels[j] = c
            j_neighbors = range_query(dataset, j, eps)
            if j_neighbors.size >= min_samples:
                seed_set.update(set(j_neighbors))

        c += 1

    return labels


def visualize_dbscan_result(dataset: np.ndarray, labels: np.ndarray) -> None:
    """可视化sklean中的DBSCAN算法"""
    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='plasma', s=15, marker='o')
    plt.show()


def main():
    # dataset = np.loadtxt('sample.txt')  # 0.5, 5
    import pandas as pd
    # dataset = pd.read_csv('./datasets/D5.csv').to_numpy()  # 0.5, 5
    # dataset = pd.read_csv('./datasets/D6.csv').to_numpy()  # 1, 5
    # dataset = pd.read_csv('./datasets/D7.csv').to_numpy()  # 6, 5
    dataset = pd.read_csv('./datasets/D8.csv').to_numpy()  # 9, 5
    # dataset = pd.read_csv('./datasets/D9.csv').to_numpy()  # 0.90, 5
    # dataset = pd.read_csv('./datasets/D10.csv').to_numpy()  # 0.15, 5
    # dataset = pd.read_csv('./datasets/D11.csv').to_numpy()  # 0.15, 5
    # dataset = pd.read_csv('./datasets/D12.csv').to_numpy()  # 0.5, 5

    eps, min_samples = 9, 5

    # 使用sklearn中的DBSCAN算法
    # noinspection PyUnresolvedReferences
    labels_std = DBSCAN(eps=eps, min_samples=min_samples).fit(dataset).labels_
    print(np.unique(labels_std))

    # 使用自己实现的DBSCAN算法
    labels = dbscan(dataset, eps=eps, min_samples=min_samples)
    print(np.unique(labels))

    # 可视化聚类结果
    visualize_dbscan_result(dataset, labels_std)
    visualize_dbscan_result(dataset, labels)


if __name__ == '__main__':
    main()
