import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances


def range_query(data: np.ndarray, idx: int, epsilon: float) -> np.ndarray:
    """获取样本点i的epsilon邻域"""
    distances = pairwise_distances(data, data[[idx]])
    judge_distances = (distances <= epsilon)
    return np.where(judge_distances.flatten() == 1)[0]


def dbscan(dataset: np.ndarray, eps: float, min_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    来自英文维基百科的DBSCAN算法
    dbscan聚类有三种类型的点
    1. 核心点：密度大于等于min_samples的点
    2. 噪声点：密度小于min_samples，同时不是任何一个核心点的邻域点
    3. 边界点：密度小于min_samples，同时是某一个核心点的邻域点
    聚类结果：每一个簇由若干个核心点和边界点组成，噪声点不属于任何簇，噪声点的标签为-1
    """

    # 求出样本点数量
    n_samples = len(dataset)
    # 初始化核心点索引
    core_indices = list()

    # 初始化所有点的标签为-2(未访问)
    labels = np.full(shape=n_samples, fill_value=-2)
    c = -1
    for i in range(n_samples):
        if labels[i] != -2:
            # 如果i点已经被访问，跳过
            continue

        # 获取样本点i的eps邻域
        neighbors = range_query(dataset, i, eps)

        if neighbors.size < min_samples:
            # 若样本点i是非核心点（包括边界点和噪声点）
            labels[i] = -1  # 标签设置为-1
            continue  # i点处理完毕，处理下一个样本点
        else:
            # 若样本点i是核心点，则创建一个新簇，将当前样本点i加入这个簇
            c += 1
            labels[i] = c
            core_indices.append(i)

            # 将样本点i邻域内的其他点加入种子集合
            seed_set = set(np.delete(neighbors, np.where(neighbors == i)[0]))

            # 处理样本点i所有的邻域点
            while seed_set:
                j = seed_set.pop()

                if labels[j] != -2:
                    # 如果j被访问过了，j可能是核心点或者非核心点
                    if labels[j] == -1:
                        # j之前被访问过且被标记为非核心点，j又是核心点i的邻域点，因此j是边界点而不是噪声点，并将j加入当前簇
                        labels[j] = c
                    else:
                        # j之前被访问过而且是核心点，则继续处理i的下一个邻域点
                        continue
                else:
                    # 若j之前未被访问，由于i是核心点且j是i的邻域点，因此将j加入当前的簇
                    labels[j] = c
                    j_neighbors = range_query(dataset, j, eps)

                    # 进一步判断j是否是核心点，如果是核心点，那么将j的邻域点加入种子集合
                    if j_neighbors.size >= min_samples:
                        seed_set.update(set(j_neighbors))
                        core_indices.append(j)

    return labels, np.array(core_indices)


def visualize_dbscan_result(dataset: np.ndarray, labels: np.ndarray) -> None:
    """可视化sklean中的DBSCAN算法"""
    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='plasma', s=5, marker='o')
    plt.title("DBSCAN")
    plt.show()


def main():
    dataset = np.loadtxt('sample.txt')  # 0.3, 5
    # import pandas as pd
    # dataset = pd.read_csv('./datasets/D5.csv').to_numpy()  # 0.5, 5
    # dataset = pd.read_csv('./datasets/D6.csv').to_numpy()  # 1, 5
    # dataset = pd.read_csv('./datasets/D7.csv').to_numpy()  # 6, 5
    # dataset = pd.read_csv('./datasets/D8.csv').to_numpy()  # 9, 5
    # dataset = pd.read_csv('./datasets/D9.csv').to_numpy()  # 0.90, 5
    # dataset = pd.read_csv('./datasets/D10.csv').to_numpy()  # 0.15, 5
    # dataset = pd.read_csv('./datasets/D11.csv').to_numpy()  # 0.15, 5
    # dataset = pd.read_csv('./datasets/D12.csv').to_numpy()  # 0.5, 5

    eps, min_samples = 0.3, 5

    # 使用sklearn中的DBSCAN算法
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dataset)

    # 获取样本点的标签
    labels = clustering.labels_
    visualize_dbscan_result(dataset, labels)
    print(np.unique(labels))

    # 获取所有点的索引集合
    all_indices = np.array(range(len(dataset)))

    # 获取核心点的索引集合
    core_indices = clustering.core_sample_indices_
    # 获取噪声点的索引集合
    noise_indices = np.where(labels == -1)[0]
    # 获取边界点的索引集合
    border_indices = np.array(list(set(all_indices) - set(core_indices) - set(noise_indices)))

    # 使用自定义的dbscan函数
    my_labels, my_core_indices = dbscan(dataset, eps, min_samples)
    # 获取噪声点的索引集合
    my_noise_indices = np.where(my_labels == -1)[0]
    # 获取边界点的索引集合
    my_border_indices = np.array(list(set(all_indices) - set(my_core_indices) - set(my_noise_indices)))

    # 判断自定义的dbscan是否相等
    print(np.all(my_labels == labels))
    print(set(core_indices) == set(my_core_indices))
    print(set(noise_indices) == set(my_noise_indices))
    print(set(border_indices) == set(my_border_indices))


if __name__ == '__main__':
    main()
