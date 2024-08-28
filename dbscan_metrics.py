import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt


def calculate_clustering_metrics(data: np.ndarray, labels: np.ndarray) -> None:
    # 确保至少有两个聚类才能计算指标
    if len(np.unique(labels)) >= 2:
        silhouette = silhouette_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    else:
        print("Not enough clusters to calculate metrics.")


def calculate_sample_numbers_of_each_cluster(labels: np.ndarray) -> None:
    # 计算每个聚类的包含的样本点个数(熟悉np.unique函数)
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"Noise points: {count}")
        else:
            print(f"Cluster {label}: {count} points")


def visualize_dbscan_clustering_results(data: np.ndarray, labels: np.ndarray) -> None:
    # 可视化聚类结果，包括噪声点
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title('DBSCAN Clustering Results')

    # 标记噪声点
    noise_mask = (labels == -1)
    plt.scatter(data[noise_mask, 0], data[noise_mask, 1], c='red', marker='x', s=100, label='Noise Points')
    plt.legend()
    plt.show()


def main():
    # make_blobs: Generate isotropic Gaussian blobs for clustering.
    data, true_labels = make_blobs(n_samples=300, n_features=2, centers=4, cluster_std=0.60, random_state=0)

    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(data)

    # # 计算内部聚类指标(排除噪声点)
    # valid_points = labels != -1
    # valid_data = data[valid_points]
    # valid_labels = labels[valid_points]

    # 可视化DBSCAN聚类结果
    visualize_dbscan_clustering_results(data, labels)

    # 计算DBSCAN聚类指标
    calculate_clustering_metrics(data, labels)

    # 计算每一个簇包含的样本点个数
    calculate_sample_numbers_of_each_cluster(labels)


if __name__ == '__main__':
    main()
