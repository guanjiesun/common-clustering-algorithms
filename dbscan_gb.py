import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from gbs import GranularBall
from gbs import generate_gbs
from gbs import verify_gbs
from gbs import visualize_gbs
from dbscan import visualize_dbscan_result as visualize_gb_dbscan_result


def visualize_original_data(data: np.ndarray) -> None:
    """可视化原始数据"""
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s=5, marker='.', color='black')
    ax.set_title('Original Data')
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def visualize_gbs_centroids(gbs: list[GranularBall], gb_labels: np.ndarray) -> None:
    """可视化粒球空间：只绘制粒球的质心，不同簇标签的粒球质心显示不同的颜色"""
    fig, ax = plt.subplots()
    n_gb = len(gbs)
    gb_centers = [gbs[i].centroid for i in range(n_gb)]
    gb_centers = np.array(gb_centers)
    # 绘制粒球的的质心
    ax.scatter(gb_centers[:, 0], gb_centers[:, 1], c=gb_labels, cmap='viridis', marker=',', s=5)
    ax.set_title("Granular Balls Without Circles")
    plt.show()


def get_sample_labels(gb_labels: np.ndarray, gbs) -> np.ndarray:
    """获取数据集中每一个样本的簇标签"""
    n_samples = 0
    for gb in gbs:
        # 计算数据集样本点个数，用于初始化sample_labels
        n_samples += len(gb.indices)

    # 初始化每一个样本的簇标签为-1
    sample_labels = np.full(n_samples, -1)
    for i, gb in enumerate(gbs):
        for sample_idx in gb.indices:
            # 样本的簇标签和它所属的粒球的簇标签保持一致
            sample_labels[sample_idx] = gb_labels[i]

    return sample_labels


def main() -> None:
    """
    1. 生成粒球空间
    2. 对粒球空间进行DBSCAN聚类，同时得到每一个粒球的簇标签
    3. 获取每一个样本点的簇标签，然后可视化GB-DBSCAN聚类结果
    """
    dataset = np.loadtxt('sample.txt')  # 加载数据集
    gbs = generate_gbs(dataset)  # 生成粒球空间
    verify_gbs(gbs)  # 验证粒球空间的有效性
    visualize_original_data(dataset)  # 可视化原始数据
    visualize_gbs(gbs)  # 可视化粒球空间

    # 对粒球空间进行DBSCAN聚类，然后获取每一个粒球的簇标签
    gb_centers = np.array([gbs[i].centroid for i in range(len(gbs))])
    # noinspection PyUnresolvedReferences
    gb_labels = DBSCAN(eps=0.85, min_samples=6).fit(gb_centers).labels_

    # 可视化粒球空间（只绘制每一个粒球的质心）
    visualize_gbs_centroids(gbs, gb_labels)

    # 基于粒球簇标签，可以获取样本点的簇标签
    sample_labels = get_sample_labels(gb_labels, gbs)
    print(np.unique(sample_labels))

    # 可视化GB-DBSCAN的聚类结果
    visualize_gb_dbscan_result(dataset, sample_labels)


if __name__ == '__main__':
    main()
