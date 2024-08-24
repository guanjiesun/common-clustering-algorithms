import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


def range_query(data: np.ndarray, idx: int, epsilon: float) -> np.ndarray:
    """获取样本点i的epsilon邻域"""
    distances = pairwise_distances(data, data[[idx]])
    judge_distances = (distances <= epsilon)
    return np.where(judge_distances.flatten() == 1)[0]


def dbscan(dataset: np.ndarray, eps: float, min_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    DBSCAN算法维基百科简介(https://en.wikipedia.org/wiki/DBSCAN)
    DBSCAN算法原始论文(https://dl.acm.org/doi/10.5555/3001460.3001507)
    DBSCAN聚类有三种类型的点
    1. 核心点：密度 >= min_samples
    2. 噪声点：密度 = 1，邻域中只有自己
    3. 边界点：1 < 密度 < min_samples，处于某一个核心点的邻域中
    聚类结果：每一个簇由若干个核心点和边界点组成，噪声点不属于任何簇，噪声点的标签为-1
    """

    # 求出样本点数量
    n_samples = len(dataset)

    # 保存核心点(保存核心点在数据集中的索引)
    core_points = list()

    # 所有样本点刚开始都没有被访问过
    flags = [False for _ in range(n_samples)]

    # 初始化样本点的簇标签，表示所有的点刚开始都是非核心点
    labels = np.full(n_samples, -1)
    c = -1
    for i in range(n_samples):
        if flags[i] is True:
            continue  # 如果i点已经被访问，继续处理数据集中的下一个样本点

        # 之后一定会访问样本点i，因此标记i已被访问
        flags[i] = True
        # 获取样本点i的eps邻域
        neighbors = range_query(dataset, i, eps)

        # 判断样本点i是核心点还是非核心点
        if neighbors.size < min_samples:
            # 若样本点i是非核心点, 则i标记为已访问，簇标签保持-1不变，然后继续处理下一个样本点
            continue
        elif neighbors.size >= min_samples:
            # 若样本点i是核心点，则创建一个新簇，将i加入当前的簇
            c += 1
            labels[i] = c
            core_points.append(i)

            """将核心点i的邻域内的点加入种子集合
            样本点i的邻域中的样本点有四种情况：
            1. j已被访问且被标记为非核心点
            2. j已被访问且被标记为核心点
            若j未被访问，那么j至少是一个边界点
            3. j是边界点，则簇标签和i保持一致，然后继续处理i的下一个邻域点
            4. j是核心点，则簇标签和i保持一致，然后将j的邻域点也加入种子集合
            """
            seed_set = set(np.delete(neighbors, np.where(neighbors == i)[0]))

            # 处理种子集合中所有的样本点
            while seed_set:
                j = seed_set.pop()
                if flags[j] is True:
                    # 如果j被访问过了，那么j有以下两种可能：j是核心点或者j是非核心点
                    if labels[j] == -1:
                        # j之前被访问过且被标记为非核心点，j又是核心点i的邻域点，因此j是边界点而不是噪声点，所以将j加入当前簇
                        labels[j] = c
                    else:
                        # j之前被访问过而且被标记为核心点，则继续处理种子集合中的下一个样本点
                        continue
                else:
                    # 若j未被访问，由于i是核心点且j是i的邻域点，因此将j加入当前的簇，并且标记j已被访问
                    labels[j] = c
                    flags[j] = True

                    # 进一步判断j是否是核心点，如果是，那么将j的邻域点加入种子集合
                    j_neighbors = range_query(dataset, j, eps)
                    if j_neighbors.size >= min_samples:
                        seed_set.update(set(j_neighbors))
                        core_points.append(j)

    return labels, np.array(core_points)


def visualize_dbscan_result(dataset: np.ndarray, labels: np.ndarray) -> None:
    """可视化DBSCAN聚类结果"""
    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='plasma', s=5, marker='o')
    plt.title("DBSCAN")
    plt.show()


def main():
    # import pandas as pd
    # dataset = pd.read_csv('./datasets/D1.csv').to_numpy()  # 0.3, 5
    # dataset = pd.read_csv('./datasets/D2.csv').to_numpy()  # 3, 6
    # dataset = pd.read_csv('./datasets/D3.csv').to_numpy()  # 0.3, 5
    # dataset = pd.read_csv('./datasets/D4.csv').to_numpy()  # 0.1, 5
    # dataset = pd.read_csv('./datasets/D5.csv').to_numpy()  # 0.5, 5
    # dataset = pd.read_csv('./datasets/D6.csv').to_numpy()  # 1, 5
    # dataset = pd.read_csv('./datasets/D7.csv').to_numpy()  # 6, 5
    # dataset = pd.read_csv('./datasets/D8.csv').to_numpy()  # 9, 5
    # dataset = pd.read_csv('./datasets/D9.csv').to_numpy()  # 0.64, 14
    # dataset = pd.read_csv('./datasets/D10.csv').to_numpy()  # 0.15, 5
    # dataset = pd.read_csv('./datasets/D11.csv').to_numpy()  # 0.15, 5
    # dataset = pd.read_csv('./datasets/D12.csv').to_numpy()  # 0.5, 5

    # 加载数据集
    dataset = np.loadtxt('sample.txt')  # 0.3, 5
    eps, min_samples = 0.3, 5

    # 获取样本点的标签
    my_labels, my_core_indices = dbscan(dataset, eps, min_samples)
    print(np.unique(my_labels))

    # 可视化DBSCAN的聚类结果
    visualize_dbscan_result(dataset, my_labels)

    # # 使用sklearn中的DBSCAN算法
    # from sklearn.cluster import DBSCAN
    # clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dataset)
    #
    # # 获取样本点的标签
    # labels = clustering.labels_
    # visualize_dbscan_result(dataset, labels)
    # print(np.unique(labels))
    #
    # # 获取所有点的索引集合
    # all_indices = np.array(range(len(dataset)))
    # # 获取核心点的索引集合
    # core_indices = clustering.core_sample_indices_
    # # 获取噪声点的索引集合
    # noise_indices = np.where(labels == -1)[0]
    # # 获取边界点的索引集合
    # border_indices = np.array(list(set(all_indices) - set(core_indices) - set(noise_indices)))
    #
    # # 使用自定义的dbscan函数
    # my_labels, my_core_indices = dbscan(dataset, eps, min_samples)
    # # 获取噪声点的索引集合
    # my_noise_indices = np.where(my_labels == -1)[0]
    # # 获取边界点的索引集合
    # my_border_indices = np.array(list(set(all_indices) - set(my_core_indices) - set(my_noise_indices)))
    #
    # # 判断自定义的dbscan是否相等
    # print(np.all(my_labels == labels))
    # print(set(core_indices) == set(my_core_indices))
    # print(set(noise_indices) == set(my_noise_indices))
    # print(set(border_indices) == set(my_border_indices))


if __name__ == '__main__':
    main()
