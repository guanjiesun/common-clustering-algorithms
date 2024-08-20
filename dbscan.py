import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances


def range_query(dataset: np.ndarray, i: int, epsilon: float) -> np.ndarray:
    """获取样本点i的epsilon邻域"""
    distances = pairwise_distances(dataset, dataset[[i]])
    judge_distances = (distances <= epsilon)
    return np.where(judge_distances.flatten() == 1)[0]


def dbscan(dataset: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    动手实现DBSCAN算法(参考英文维基百科)
    dbscan聚类结果产生三种类型的点
    1. 核心点：密度大于等于min_samples的点
    2. 噪声点：密度小于min_samples，同时不是任何一个核心点的邻域点
    3. 边界点：密度小于min_samples，同时是某一个核心点的邻域点
    """
    n_samples = len(dataset)

    # 标签为-2，表示此样本点未被访问过
    labels = np.full(shape=n_samples, fill_value=-2)
    c = 0
    for i in range(n_samples):
        if labels[i] != -2:
            continue  # 如果点已经被处理，跳过

        neighbors = range_query(dataset, i, eps)  # 获取样本点i的eps邻域

        if neighbors.size < min_samples:
            labels[i] = -1  # 非核心点（包括边界点和噪声点）的的标签设置为-1，然后跳过处理下一个点
            continue

        # 若样本点i是核心点，则运行以下代码
        labels[i] = c

        # 获取样本点i的种子集合(i所有的邻域点，但是不包含i)
        seed_set = set(np.delete(neighbors, np.where(neighbors == i)[0]))

        # 处理样本点i所有的邻域点
        while seed_set:
            j = seed_set.pop()
            if labels[j] == -1:
                labels[j] = c  # 让j的簇标签和核心点i的簇标签保持一致
            if labels[j] != -2:
                continue  # 如果j已经被处理过，则跳过循环，继续处理样本点i的其他邻域点

            labels[j] = c
            j_neighbors = range_query(dataset, j, eps)
            if j_neighbors.size >= min_samples:
                # 如果j也是核心点，那么将j的邻域点加入种子集合
                seed_set.update(set(j_neighbors))

        c += 1

    return labels


def visualize_dbscan_result(dataset: np.ndarray, labels: np.ndarray) -> None:
    """可视化sklean中的DBSCAN算法"""
    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='plasma', s=15, marker='o')
    plt.show()


def main():
    dataset = np.loadtxt('sample.txt')  # 0.5, 5
    # import pandas as pd
    # dataset = pd.read_csv('./datasets/D5.csv').to_numpy()  # 0.5, 5
    # dataset = pd.read_csv('./datasets/D6.csv').to_numpy()  # 1, 5
    # dataset = pd.read_csv('./datasets/D7.csv').to_numpy()  # 6, 5
    # dataset = pd.read_csv('./datasets/D8.csv').to_numpy()  # 9, 5
    # dataset = pd.read_csv('./datasets/D9.csv').to_numpy()  # 0.90, 5
    # dataset = pd.read_csv('./datasets/D10.csv').to_numpy()  # 0.15, 5
    # dataset = pd.read_csv('./datasets/D11.csv').to_numpy()  # 0.15, 5
    # dataset = pd.read_csv('./datasets/D12.csv').to_numpy()  # 0.5, 5

    eps, min_samples = 0.5, 5

    # 使用sklearn中的DBSCAN算法
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dataset)

    # noinspection PyUnresolvedReferences
    labels = clustering.labels_
    visualize_dbscan_result(dataset, labels)
    print(np.unique(labels))

    # 获取核心点，边界点和噪声点
    all_indices = set(range(len(dataset)))  # 获取所有点的索引集合
    # noinspection PyUnresolvedReferences
    core_indices = set(clustering.core_sample_indices_)  # 获取核心点的索引集合
    noise_indices = set(np.where(labels == -1)[0])  # 获取噪声点的索引集合
    border_indices = all_indices - core_indices - noise_indices  # 使用集合运算获取边界点的索引集合
    print(len(all_indices), len(core_indices), len(noise_indices), len(border_indices))


if __name__ == '__main__':
    main()
