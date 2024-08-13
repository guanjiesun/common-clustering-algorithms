from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.metrics import pairwise_distances

from visualize_original_data import visualize_original_data


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
    """将一个粒球gb划分为两个粒球gb_child1和gb_child2的K-Means++聚类算法"""

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
        else:
            # 如果gb大小合适，则将此gb加入gbs
            gbs.append(gb)

    return gbs


def verify_gbs(gbs: list[GranularBall]) -> None:
    """利用Python中集合的特性，验证生成的粒球空间的正确性"""
    size = len(gbs)

    # 任意两个粒球的indices互不相交，则生成的粒球空间正确
    for i in range(size):
        set_i = set(gbs[i].indices)
        for j in range(i+1, size):
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
        ax.scatter(gb.centroid[0], gb.centroid[1], color='red', s=5)
        ax.scatter(gb.data[:, 0], gb.data[:, 1], color='black', marker='.', s=5)
        ax.set_title("Granular Ball Space")
        ax.set_aspect('equal', adjustable='box')

    plt.show()


def main() -> None:
    """
    1. 基于一个给定的数据集，生成粒球空间
    2. 验证粒球空间的正确性
    3. 可视化粒球空间
    """

    # 载入数据
    dataset_path = Path('./datasets_from_gbsc/D1.csv')
    dataset = pd.read_csv(dataset_path, header=None).to_numpy()

    # 生成粒球空间
    gbs = generate_gbs(dataset)

    # 验证粒球空间的有效性
    verify_gbs(gbs)

    # 可视化原始数据
    visualize_original_data(dataset)

    # 可视化粒球空间
    visualize_gbs(gbs)


if __name__ == '__main__':
    main()
