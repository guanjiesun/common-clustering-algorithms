from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.metrics import pairwise_distances

from kmeans import visualize_original_data
from dp import calculate_delta as calculate_gb_delta
from dp import generate_decision_graph
from dp import assign_points_to_clusters as assign_gb_to_clusters


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


def visualize_gbs_centroids(gbs: list[GranularBall], gb_centroids: list[int], gb_labels: np.ndarray) -> None:
    """可视化粒球空间，只绘制粒球的质心，和作为聚类中心的粒球的质心"""
    fig, ax = plt.subplots()
    n_gb = len(gbs)
    gb_centers = [gbs[i].centroid for i in range(n_gb)]
    gb_centers = np.array(gb_centers)

    # 绘制非聚类中心粒球的的质心
    ax.scatter(gb_centers[:, 0], gb_centers[:, 1], c=gb_labels, cmap='viridis', marker=',', s=5)

    for centroid_idx in gb_centroids:
        # 绘制作为聚类中心的粒球的质心
        centroid = gbs[centroid_idx].centroid
        # 只能绘制平面上的图
        ax.scatter(centroid[0], centroid[1], color='red', marker='*', s=20)

    ax.set_title("Granular Balls Without Circles")

    plt.show()


def calculate_gb_rho(gbs: list[GranularBall]):
    """计算每一个粒球的局部密度(local density, rho)"""
    n_gb = len(gbs)
    rho = np.zeros(n_gb)
    for i, gb in enumerate(gbs):
        size = gb.size
        radius = gb.radius
        centroid = gb.centroid
        data = gb.data
        if size == 1:
            # 如果粒球gb只包含一个样本点，那么它的半径为0，它的局部密度设置为1
            rho[i] = 1
            continue
        foo = np.sum(pairwise_distances(data, centroid.reshape(1, -1)))
        bar = np.square(size/radius)
        if radius == 0 or foo == 0:
            raise ZeroDivisionError("Check Your Code!")
        rho[i] = bar / foo

    return rho


def distances_matrix(gbs: list[GranularBall]) -> np.ndarray:
    """基于粒球空间，计算每一对粒球之间的欧式距离"""
    n_gbs = len(gbs)
    distances = np.zeros((n_gbs, n_gbs))
    for i in range(n_gbs-1):
        for j in range(i+1, n_gbs):
            # distance表示gbs[]和gbs[j]这两个粒球之间的距离
            distance = pairwise_distances(gbs[i].centroid.reshape(1, -1), gbs[j].centroid.reshape(1, -1))
            distance = distance.flatten()[0]
            distances[i, j] = distance
            distances[j, i] = distance

    return distances


def assign_sample_to_clusters(dataset: np.ndarray, labels: np.ndarray, gbs) -> np.ndarray:
    """获取每一个样本的簇标签"""
    n_samples = len(dataset)
    sample_labels = np.full(n_samples, -1)
    for i, gb in enumerate(gbs):
        for sample_idx in gb.indices:
            # 样本的簇标签和它所属的粒球的簇标签保持一致
            sample_labels[sample_idx] = labels[i]

    return sample_labels


def visualize_gbdp_clustering(dataset: np.ndarray, sample_labels: np.ndarray) -> None:
    fig, ax = plt.subplots()
    # 绘制样本点
    ax.scatter(dataset[:, 0], dataset[:, 1], c=sample_labels, marker='.', s=5)
    ax.set_title("GBDP Clustering Result")
    plt.show()


def main() -> None:
    """
    1. 基于一个给定的数据集，生成粒球空间
    2. 验证粒球空间的正确性
    3. 可视化粒球空间
    """
    # folder_path = Path('./datasets_from_gbsc')
    # dataset_paths = list(folder_path.glob("*.csv"))

    # dataset, np.ndarray, shape=(n_sample, m_features)
    dataset_path = Path('./datasets_from_gbsc/D3.csv')
    dataset = pd.read_csv(dataset_path).to_numpy()
    # dataset = np.loadtxt(dataset_path)

    # 生成粒球空间
    gbs = generate_gbs(dataset)

    # 验证粒球空间的有效性
    verify_gbs(gbs)

    # 可视化粒球空间
    visualize_original_data(dataset)
    visualize_gbs(gbs)

    # 基于粒球空间，计算粒球距离矩阵
    distances = distances_matrix(gbs)

    # 计算每一个粒球的局部密度
    rho = calculate_gb_rho(gbs)

    # 计算每一个粒球的delta距离和最近邻
    delta, nearest_neighbor = calculate_gb_delta(distances, rho)

    # 生成决策图并选取聚类中心
    gb_centroids = generate_decision_graph(rho, delta)

    # 获取每一个粒球的簇标签
    gb_labels = assign_gb_to_clusters(rho, gb_centroids, nearest_neighbor)

    # 可视化粒球空间，每一个粒球不显示圆心，不同簇标签的粒球显示不同的颜色
    visualize_gbs_centroids(gbs, gb_centroids, gb_labels)

    # 获取每一个样本的簇标签
    sample_labels = assign_sample_to_clusters(dataset, gb_labels, gbs)

    # 可视化GBDPC聚类结果
    visualize_gbdp_clustering(dataset, sample_labels)


if __name__ == '__main__':
    main()
