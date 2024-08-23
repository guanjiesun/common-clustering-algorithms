import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from gbs import GranularBall
from gbs import generate_gbs
from gbs import verify_gbs
from gbs import visualize_gbs

from dbscan_3w import get_three_way_dbscan_result as get_3w_gb_dbscan_result


def visualize_original_data(data: np.ndarray) -> None:
    """可视化原始数据"""
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s=5, marker='.', color='black')
    ax.set_title('Original Data')
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def visualize_gb_dbscan_result_gbs(gbs: list[GranularBall], gb_labels: np.ndarray) -> None:
    """可视化GB-DBSCAN聚类结果(粒球视角)"""
    fig, ax = plt.subplots()
    n_gb = len(gbs)
    gb_centroids = [gbs[i].centroid for i in range(n_gb)]
    gb_centroids = np.array(gb_centroids)
    # 绘制粒球的的质心
    ax.scatter(gb_centroids[:, 0], gb_centroids[:, 1], c=gb_labels, cmap='plasma', marker='o', s=10)
    ax.set_title("GB-DBSCAN (GB Viewpoint)")
    plt.show()


def get_sample_labels(gb_labels: np.ndarray, gbs) -> np.ndarray:
    """GB-DBSCAN聚类之后，获取样本点的簇标签"""
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


def visualize_gb_dbscan_result_samples(dataset: np.ndarray, sample_labels: np.ndarray) -> None:
    """可视化GB-DBSCAN聚类结果(样本点视角)"""
    plt.scatter(dataset[:, 0], dataset[:, 1], c=sample_labels, cmap='plasma', s=5, marker='.')
    plt.title("GB-DBSCAN (Sample Viewpoint)")
    plt.show()


def reassign_gb_labels(gb_labels: np.ndarray, clusters: list) -> np.ndarray:
    """基于GB-DBSCAN的结果，进行三支决策之后：正域中的粒球分配相应的簇标签，边界中的粒球统一分配的簇标签是-1"""
    k = len(clusters)
    new_gb_labels = np.full(len(gb_labels), -1)

    # 给正域中的粒球分配簇标签，边界域中的粒球保持为-1即可
    for i in range(k):
        cluster = clusters[i]
        for pos_gb_idx in cluster[0]:
            new_gb_labels[pos_gb_idx] = i

    return new_gb_labels


def reassign_sample_labels(dataset, clusters, new_gb_labels, gbs: list[GranularBall]) -> np.ndarray:
    """基于GB-DBSCAN的结果，进行三支决策之后：样本点和它们所属粒球的簇标签保持一致"""
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
    plt.title("3W-GB-DBSCAN (GB Viewpoint)")
    plt.show()


def visualize_3w_gb_dbscan_samples(dataset: np.ndarray, sample_labels: np.ndarray):
    """可视化3W-GB-DBSCAN的聚类结果(样本点视角)"""
    plt.scatter(dataset[:, 0], dataset[:, 1], c=sample_labels, cmap='plasma', s=5, marker='.')
    plt.title("3W-GB-DBSCAN (Samples Viewpoint)")
    plt.show()


def main() -> None:
    """
    1. 生成粒球空间
    2. 进行GB-DBSCAN聚类，得到粒球的簇标签和样本点的簇标签
    3. 可视化GB-DBSCAN的聚类结果
    4. 基于GB-DBSCAN的聚类结果，利用三支决策，将核心粒球、边界粒球和噪声粒球分配到相应簇的核心域或者边界域
    5. 基于三支的结果，给正域的粒球和边界域的粒球重新分配簇标签，样本点也是如此
    6. 可视化3W-GB-DBSCAN的聚类结果
    """

    # TODO 基于一个数据集，生成粒球空间
    dataset = np.loadtxt('sample.txt')  # 加载数据集
    gbs = generate_gbs(dataset)  # 生成粒球空间
    verify_gbs(gbs)  # 验证粒球空间的有效性
    visualize_original_data(dataset)  # 可视化原始数据
    visualize_gbs(gbs)  # 可视化粒球空间

    # TODO 对粒球进行GB-DBSCAN聚类，得到三种类型的粒球：核心粒球、边界粒球和噪声粒球
    gb_centroids = np.array([gbs[i].centroid for i in range(len(gbs))])
    eps, min_samples = 0.75, 6
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(gb_centroids)
    # 核心粒球、边界粒球的簇标签是0, 1, 2..., 噪声粒球的簇标签是-1, 噪声粒球可能不存在
    gb_labels = clustering.labels_
    sample_labels = get_sample_labels(gb_labels, gbs)
    print(np.unique(gb_labels))
    print(np.unique(sample_labels))

    # 可视化GB-DBSCAN聚类结果(粒球视角, 样本点视角)
    visualize_gb_dbscan_result_gbs(gbs, gb_labels)
    visualize_gb_dbscan_result_samples(dataset, sample_labels)

    # TODO 基于GB-DBSCAN的聚类结果，进行三支决策
    # 获取核心粒球、边界粒球和噪声粒球
    all_indices = np.array(range(len(gbs)))
    core_indices = clustering.core_sample_indices_
    noise_indices = np.where(gb_labels == -1)[0]
    border_indices = np.array(list(set(all_indices) - set(core_indices) - set(noise_indices)))

    # 利用三支决策，将核心粒球、边缘粒球和噪声粒球分配到相应簇的正域或者边界域
    clusters = get_3w_gb_dbscan_result(gb_centroids, gb_labels, core_indices, noise_indices, border_indices, eps)

    # 给正域的粒球和边界域的粒球分配相应的簇标签(边界域的粒球簇标签统一设置为-1)
    new_gb_labels = reassign_gb_labels(gb_labels, clusters)
    # 基于粒球的簇标签new_gb_labels，获取样本点的簇标签
    new_sample_labels = reassign_sample_labels(dataset, clusters, new_gb_labels, gbs)
    print(np.unique(new_gb_labels))
    print(np.unique(new_sample_labels))

    # 可视化3W-GB-DBSCAN聚类结果(粒球视角, 样本点视角)
    visualize_3w_gb_dbscan_gbs(gb_centroids, new_gb_labels)
    visualize_3w_gb_dbscan_samples(dataset, new_sample_labels)


if __name__ == '__main__':
    main()
