from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.metrics import pairwise_distances


class GranularBall:
    def __init__(self, dataset: np.ndarray, indices: np.ndarray):
        """
        TODO 粒球的属性indices存储属于粒球的数据点在原始数据集data中的索引
        TODO indices是一个粒球最根本最重要的属性
        TODO indices: np.ndarray, shape=(len(indices))
        """
        self.indices = indices
        self.data = dataset[indices]
        self.size = len(self.data)
        self.centroid = np.mean(self.data, axis=0)
        self.radius = np.max(pairwise_distances(self.data, self.centroid.reshape(1, -1)))


def kmeans(dataset: np.ndarray, gb: GranularBall, k: int = 2, max_iterations: int = 100,
           tolerance: float = 1e-4) -> tuple[GranularBall, GranularBall]:
    """将一个粒球gb划分为两个粒球gb_child1和gb_child2的K-Means聚类算法"""

    # 获取gb的属性
    indices, data, size = gb.indices, gb.data, gb.size

    # 初始化k个聚类中心
    np.random.seed(size)
    centroid_idx = np.random.choice(size, k, replace=False)
    # centroids: np.ndarray, shape = (k, m_features)
    centroids = data[centroid_idx]

    # 开始迭代
    n_iteration = 0
    while True:
        n_iteration += 1

        # distances: np.ndarray, shape=(k, size), 表示每一个聚类中心到其他所有点的距离
        distances = pairwise_distances(centroids, data)

        # labels保存每一个样本的簇标签
        labels = np.argmin(distances, axis=0)

        # clusters保存每一个簇的数据点
        clusters = list()
        for i in range(k):
            # indices的作用：确保cluster保存的数据点的索引是在原始数据集中的索引
            # TODO 一定要保证cluster保存的数据点的索引是在原始数据集中dataset中的索引
            cluster = indices[np.where(labels == i)[0]]
            clusters.append(cluster)

        # 计算新的聚类中心new_centroids: np.ndarray, shape=(k, m_features)
        new_centroids = np.array([np.mean(data[labels == i], axis=0) for i in range(k)])

        # TODO 阈值tolerance用于判断算法是否收敛
        if np.all(np.abs(new_centroids-centroids) < tolerance) or n_iteration > max_iterations:
            break

        # 若算法未收敛，则更新聚类中心，继续迭代
        centroids = new_centroids

    # 求出粒球的划分
    gb_child1 = GranularBall(dataset, clusters[0])
    gb_child2 = GranularBall(dataset, clusters[1])

    return gb_child1, gb_child2


def generate_gbs(dataset: np.ndarray) -> list[GranularBall]:
    """基于数据集dataset，生成粒球空间gbs"""

    # 初始化gbs
    gbs = list()

    # n表示原始数据集dataset的样本数量
    n = len(dataset)

    # TODO 判断粒球是否需要进一步划分为两个小粒球的阈值(设置方式参考夏书银GB-DPC论文)
    threshold = np.sqrt(n)

    # 初始化队列(deque函数需要的参数是一个可迭代对象)
    queue = deque([GranularBall(dataset, np.arange(n))])

    # 开始生成gbs
    while queue:
        # 使用popleft而不是pop是因为要保持queue先进先出的特性
        gb = queue.popleft()
        if gb.size > threshold:
            # 如果gb太大，则使用2-means算法将gb划分为两个更小的粒球，然后两个小粒球入队
            queue.extend(kmeans(dataset, gb, k=2))
            # queue.extend(spilt_ball(dataset, gb))
        else:
            # 如果gb大小合适，则将此gb加入gbs
            gbs.append(gb)

    return gbs


def calculate_mgn(gbs: list[GranularBall]) -> list[set]:
    """计算每一个粒球的多粒度近邻(Multi Granularity Neighbors)"""
    # n 表示粒球的数量
    n = len(gbs)

    # gbc表示granular ball centroids，用粒球的中心表示每一个粒球
    gbc = np.array([gbs[i].centroid for i in range(len(gbs))])
    # dists是基于gbc的距离矩阵(每个粒球到其他粒球的距离：粒球质心之间的距离)
    dists = pairwise_distances(gbc)

    # 初始化k为1
    k = 1
    # nn列表表示每一个元素表示一个粒球的k近邻集合
    nn = [set() for _ in range(n)]
    # rnn列表的每一个元素表示一个粒球的反k近邻集合
    rnn = [set() for _ in range(n)]
    # mgn列表的每一个元素表示一个粒球的多粒度k近邻
    mgn = [set() for _ in range(n)]

    # numbers列表表示每一个粒球的反近邻的数量，初始化为0
    numbers = [0 for _ in range(n)]

    while k < n:
        # initial_number记录本次迭代开始之前没有反近邻的粒球的数量
        initial_number = numbers.count(0)

        for i, centroid in enumerate(gbc):
            # 计算i号粒球的kth近邻，用j表示（如k=3，则j号粒球是距离i号粒球第3近的粒球）
            j = np.argsort(dists[i, :])[k]
            j = int(j)
            # nn[i]表示i号粒球的k近邻集合，将j加入其中
            nn[i].add(j)
            # rnn[j]表示j号粒球的反k近邻集合，将i加入其中
            rnn[j].add(i)
            # j的反近邻数量+1
            numbers[j] += 1

        # current_number记录本次迭代结束之后没有反近邻的粒球数量
        current_number = numbers.count(0)
        if (current_number-initial_number) == 0 or current_number == 0:
            break
        k += 1

    for i, centroid in enumerate(gbc):
        # 计算每一个粒球的多粒度近邻集合
        mgn[i] = nn[i].intersection(rnn[i]).union([i])

    return mgn


def verify_gbs(gbs: list[GranularBall]) -> None:
    """利用Python中集合的特性，验证生成的粒球空间的正确性"""

    # n_gbs表示粒球的数量
    n_gbs = len(gbs)

    # 任意两个粒球的indices互不相交，则生成的粒球空间正确
    for i in range(n_gbs):
        set_i = set(gbs[i].indices)
        for j in range(i+1, n_gbs):
            set_j = set(gbs[j].indices)
            if set_i.intersection(set_j) != set():
                raise TypeError("Wrong Granular Ball Space")


def visualize_gbs(gbs: list[GranularBall]) -> None:
    """可视化粒球空间"""
    fig, ax = plt.subplots()

    for i, gb in enumerate(gbs):
        # float映射是为了符合Circle函数对参数数据类型的要求
        # gb.centroid[0], gb.centroid[1]说明此函数只能可视化样本特征数量为2的数据集生成的gbs
        float_x, float_y, float_radius = map(float, [gb.centroid[0], gb.centroid[1], gb.radius])

        # 创建圆, 绘制数据点, 绘制圆, 绘制圆心
        circle = Circle((float_x, float_y), float_radius, fill=False, color='blue')
        ax.add_artist(circle)
        ax.scatter(gb.centroid[0], gb.centroid[1], color='red', marker='o', s=10)
        ax.scatter(gb.data[:, 0], gb.data[:, 1], color='black', marker='.', s=5)

    ax.set_title("Granular Balls")
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def visualize_original_data(dataset: np.ndarray) -> None:
    """可视化原始数据"""
    fig, ax = plt.subplots()
    ax.scatter(dataset[:, 0], dataset[:, 1], s=5, marker='.', color='black')
    ax.set_title('Original Data')
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def main() -> None:
    """
    1. 基于给定数据集，生成粒球空间
    2. 验证粒球空间的正确性
    3. 可视化粒球空间
    """

    folder_path = Path('./datasets')
    csv_files = list(folder_path.glob("*.csv"))

    for csv_file in csv_files:
        # 载入数据
        dataset = pd.read_csv(csv_file, header=None).to_numpy()

        # 生成粒球空间
        gbs = generate_gbs(dataset)

        # 验证粒球空间的有效性
        verify_gbs(gbs)

        # 可视化原始数据
        visualize_original_data(dataset)

        # 可视化粒球空间
        visualize_gbs(gbs)

        # 计算每一个粒球的多粒度近邻(multi granularity neighbors)
        mgn = calculate_mgn(gbs)
        print(len(mgn) == len(gbs))


if __name__ == '__main__':
    main()
