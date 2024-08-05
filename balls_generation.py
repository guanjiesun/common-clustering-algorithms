from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from kmeans import visualize_original_data

# TODO data is a global variable
# TODO data, np.ndarray, ndim=2, shape=(n_sample, m_features)
# data = np.loadtxt('sample.txt')
data = pd.read_csv('./datasets_from_gbsc/D7.csv').to_numpy()


class GranularBall:
    def __init__(self, indices: np.ndarray):
        """
        粒球只有一个属性indices，用于存储属于粒球的数据点在原始数据集data中的索引
        indices: np.ndarray, ndim=1, shape=(len(indices))
        """
        self.indices = indices

    def get_gb_data(self):
        """根据indices，从原始数据集中复制属于粒球的数据点"""
        return data[self.indices]

    def get_centroid(self) -> np.ndarray:
        """粒球圆心(定义方式参考夏书银GB-DPC论文)"""
        return np.mean(self.get_gb_data(), axis=0)

    def get_radius(self) -> np.float64:
        """粒球半径(定义方式参考夏书银GB-DPC论文)"""
        return np.max(np.sqrt(np.sum(np.square(self.get_gb_data() - self.get_centroid()), axis=1)))


def kmeans(indices: np.ndarray, k: int = 2, max_iterations: int = 100,
           tolerance: float = 1e-4) -> list[np.ndarray]:
    """
    用于将一个粒球划分为两个粒球的K-Means聚类算法
    indices表示被划分的粒球的数据点在原始数据集中的索引
    """

    # 从data中复制属于gb的数据
    gb_data = data[indices]
    gb_size = len(gb_data)

    # 初始化k个聚类中心; centroids: np.ndarray, ndim=2, shape=(k, m_features)
    np.random.seed(gb_size)
    centroids = gb_data[np.random.choice(gb_size, k, replace=False)]
    n_iteration = 0
    while True:
        n_iteration += 1
        # distances: np.ndarray, shape=(k, gb_size), 表示每一个聚类中心到其他所有点的距离
        distances = np.sqrt(np.sum(np.square(gb_data-centroids[:, np.newaxis, :]), axis=2))
        # labels保存每一个样本的簇标签
        labels = np.argmin(distances, axis=0)

        # clusters保存每一个簇的数据点
        clusters = list()
        for i in range(k):
            # indices的作用：确保cluster保存的数据点的索引是在原始数据集中的索引
            # TODO 一定要保证cluster保存的数据点的索引是在原始数据集中的索引
            cluster = indices[np.where(labels == i)[0]]
            clusters.append(cluster)

        # 计算新的聚类中心; new_centroids: np.ndarray, ndim=2, shape=(k, m_features)
        new_centroids = np.array([gb_data[labels == i].mean(axis=0) for i in range(k)])
        # TODO 阈值tolerance用于判断算法是否收敛
        if np.all(np.abs(new_centroids-centroids) < tolerance) or n_iteration > max_iterations:
            # 新聚类中心和旧聚类中心相比变化很小，或者达到最大跌打次数，则停止迭代，聚类完成
            break

        # 如果算法没有收敛，则更新聚类中心，进行下一次迭代
        centroids = new_centroids

    # 求出粒球的划分即可，不需要每一个划分的中心点
    return clusters


def split_granular_ball(gb: GranularBall) -> (GranularBall, GranularBall):
    """使用2-Means算法将gb划分为两个子粒球"""
    clusters = kmeans(gb.indices, 2)
    gb_child1 = GranularBall(clusters[0])
    gb_child2 = GranularBall(clusters[1])

    return gb_child1, gb_child2


def generate_gbs() -> list[GranularBall]:
    """基于一个数据集data，生成一个粒球空间Granular Ball Space, gbs"""
    gbs = list()
    # n表示原始数据集data的数据点数量(设置方式参考夏书银GB-DPC论文)
    n = len(data)
    # TODO 判断粒球是否需要进一步划分为两个小粒球的阈值
    threshold = np.sqrt(n)
    # 初始化队列(deque函数需要的参数是一个可迭代对象)
    queue = deque([GranularBall(np.arange(n))])

    # 开始生成粒球空间
    while queue:
        gb = queue.popleft()  # 使用popleft而不是pop是因为要保持queue先进先出的特性
        if len(gb.indices) > threshold:
            # 如果gb太大，则使用2means算法将gb划分为两个更小的粒球，然后两个小粒球入队
            queue.extend(split_granular_ball(gb))
        else:
            gbs.append(gb)

    return gbs


def verify_gbs(gbs: list[GranularBall]) -> None:
    """利用Python中集合的特性，验证生成的粒球空间的正确性"""
    size = len(gbs)

    # 两两(粒球的数据索引)互不相交，则生成的粒球空间正确
    for i in range(size):
        set_i = set(gbs[i].indices)
        for j in range(i+1, size):
            set_j = set(gbs[j].indices)
            if set_i.intersection(set_j) != set():
                raise ValueError("Wrong Granular Ball Space")


def visualize_gbs(gbs: list[GranularBall], ax: plt.Axes) -> None:
    """可视化粒球空间"""
    for i, gb in enumerate(gbs):
        # 获取粒球的数据、质心和半径
        gb_data, centroid, radius = data[gb.indices], gb.get_centroid(), gb.get_radius()

        # float_x, float_y和float_radius是为了符合Circle函数对参数数据类型的要求
        float_x, float_y, float_radius = float(centroid[0]), float(centroid[1]), float(radius)

        # 创建圆, 绘制数据点, 绘制圆, 绘制圆心
        circle = Circle((float_x, float_y), float_radius, fill=False, color='blue')
        ax.add_artist(circle)
        ax.scatter(centroid[0], centroid[1], color='red', s=5)
        ax.scatter(gb_data[:, 0], gb_data[:, 1], color='black', marker='.', s=5)
        ax.set_title("Granular Ball Space")
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    1. 生成粒球空间
    2. 验证粒球空间的正确性
    3. 可视化粒球空间
    """
    gbs = generate_gbs()
    verify_gbs(gbs)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    visualize_original_data(data, ax0)
    visualize_gbs(gbs, ax1)


if __name__ == '__main__':
    main()
