import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from gbs import GranularBall
from gbs import generate_gbs
from gbs import verify_gbs
from gbs import visualize_gbs

from dbscan_gb import visualize_original_data
from dbscan_gb import visualize_gbs_centroids_after_dbscan
from dbscan_3w import get_three_way_dbscan_result


def reassign_gb_labels(gb_labels: np.ndarray, clusters: list) -> np.ndarray:
    """正域中的粒球分配相应的簇标签，负域中的粒球统一分配的簇标签是-1"""
    k = len(clusters)
    new_gb_labels = np.full(len(gb_labels), -1)

    # 给正域中的粒球分配簇标签
    for i in range(k):
        cluster = clusters[i]
        for pos_gb_idx in cluster[0]:
            new_gb_labels[pos_gb_idx] = i

    return new_gb_labels


def get_sample_labels(dataset, clusters, new_gb_labels, gbs: list[GranularBall]) -> np.ndarray:
    n_samples = len(dataset)
    k = len(clusters)
    sample_labels = np.full(n_samples, -1)

    for k in range(k):
        cluster = clusters[k]
        for gb_idx in cluster[0]:
            gb = gbs[gb_idx]
            gb_label = new_gb_labels[gb_idx]
            sample_labels[gb.indices] = gb_label

    return sample_labels


def visualize_3w_gb_dbscan_gbs(gb_centroids: np.ndarray, new_gb_labels: np.ndarray):
    """可视化3W-GB-DBSCAN的聚类结果(粒球视角)"""
    plt.scatter(gb_centroids[:, 0], gb_centroids[:, 1], c=new_gb_labels, cmap='plasma', s=10, marker='o')
    plt.title("3W-GB-DBSCAN Clustering Result (GB Viewpoint)")
    plt.show()


def visualize_3w_gb_dbscan_samples(dataset: np.ndarray, sample_labels: np.ndarray):
    """可视化3W-GB-DBSCAN的聚类结果(样本点视角)"""
    plt.scatter(dataset[:, 0], dataset[:, 1], c=sample_labels, cmap='plasma', s=10, marker='o')
    plt.title("3W-GB-DBSCAN Clustering Result (Samples Viewpoint)")
    plt.show()


def main() -> None:
    """
    1. 生成粒球空间
    2. 对粒球空间进行DBSCAN聚类，同时得到每一个粒球的簇标签
    3. 利用三支决策，将核心粒球、边界粒球和噪声粒球分配到相应簇的核心域或者边界域
    3. 获取每一个样本点的簇标签，然后可视化GB-DBSCAN聚类结果
    """
    dataset = np.loadtxt('sample.txt')  # 加载数据集
    gbs = generate_gbs(dataset)  # 生成粒球空间
    verify_gbs(gbs)  # 验证粒球空间的有效性
    visualize_original_data(dataset)  # 可视化原始数据
    visualize_gbs(gbs)  # 可视化粒球空间

    # 对粒球空间进行DBSCAN聚类，得到三种类型的粒球：核心粒球、边界粒球和噪声粒球
    gb_centroids = np.array([gbs[i].centroid for i in range(len(gbs))])
    eps, min_samples = 0.75, 6
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(gb_centroids)
    gb_labels = clustering.labels_
    print(np.unique(gb_labels))

    # 可视化粒球空间（只绘制每一个粒球的质心）
    visualize_gbs_centroids_after_dbscan(gbs, gb_labels)

    # 获取三种类型粒球
    all_indices = np.array(range(len(gbs)))
    core_indices = clustering.core_sample_indices_
    noise_indices = np.where(gb_labels == -1)[0]
    border_indices = np.array(list(set(all_indices) - set(core_indices) - set(noise_indices)))

    # 利用三支决策规则，将核心粒球、边缘粒球和噪声力求分配到相应簇的正域或者边界域
    clusters = get_three_way_dbscan_result(gb_centroids, gb_labels, core_indices, noise_indices, border_indices, eps)

    # 给正域的粒球和边界域的粒球分配相应的簇标签(边界域的粒球簇标签统一设置为-1)
    new_gb_labels = reassign_gb_labels(gb_labels, clusters)

    # 可视化3W-GB-DBSCAN聚类结果(粒球视角)
    visualize_3w_gb_dbscan_gbs(gb_centroids, new_gb_labels)

    # 基于粒球的簇标签new_gb_labels，获取样本点的簇标签
    sample_labels = get_sample_labels(dataset, clusters, new_gb_labels, gbs)

    # 可视化3W-GB-DBSCAN聚类结果(样本点视角)
    visualize_3w_gb_dbscan_samples(dataset, sample_labels)


if __name__ == '__main__':
    main()
