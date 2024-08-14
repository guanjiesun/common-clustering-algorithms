from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from gbs import GranularBall
from gbs import generate_gbs
from gbs import verify_gbs
from gbs import visualize_gbs
from dpc import calculate_delta as calculate_gb_delta
from dpc import generate_decision_graph
from dpc import assign_points_to_clusters as assign_gb_to_clusters
from visualize_original_data import visualize_original_data


def visualize_gbs_centroids(gbs: list[GranularBall], gb_centroids: list[int], gb_labels: np.ndarray) -> None:
    """可视化粒球空间：只绘制粒球的质心，和作为聚类中心的粒球的质心"""
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


def get_sample_labels(dataset: np.ndarray, labels: np.ndarray, gbs) -> np.ndarray:
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
    """GB-DPC算法实现"""

    # dataset, np.ndarray, shape=(n_sample, m_features)
    dataset_path = Path('./datasets/D1.csv')
    dataset = pd.read_csv(dataset_path, header=None).to_numpy()

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

    # 计算每一个粒球的delta和最近邻(密度比它大且距离自身最近的粒球就是此粒球的最近邻)
    delta, nearest_neighbor = calculate_gb_delta(distances, rho)

    # 生成决策图并选取若干个粒球作为聚类中心
    gb_centroids = generate_decision_graph(rho, delta)

    # 获取每一个粒球的簇标签
    gb_labels = assign_gb_to_clusters(rho, gb_centroids, nearest_neighbor)

    # 可视化粒球空间，每一个粒球不显示圆心，不同簇标签的粒球显示不同的颜色
    visualize_gbs_centroids(gbs, gb_centroids, gb_labels)

    # 获取每一个样本的簇标签
    sample_labels = get_sample_labels(dataset, gb_labels, gbs)

    # 可视化GB-DPC聚类结果
    visualize_gbdp_clustering(dataset, sample_labels)


if __name__ == '__main__':
    main()
