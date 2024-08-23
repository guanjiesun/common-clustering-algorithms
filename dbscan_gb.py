# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
#
# from gbs import GranularBall
# from gbs import generate_gbs
# from gbs import verify_gbs
# from gbs import visualize_gbs
#
#
# def visualize_original_data(data: np.ndarray) -> None:
#     """可视化原始数据"""
#     fig, ax = plt.subplots()
#     ax.scatter(data[:, 0], data[:, 1], s=5, marker='.', color='black')
#     ax.set_title('Original Data')
#     ax.set_aspect('equal', adjustable='box')
#     plt.show()
#
#
# def visualize_gb_dbscan_result_gbs(gbs: list[GranularBall], gb_labels: np.ndarray) -> None:
#     """可视化GB-DBSCAN聚类结果(粒球视角)"""
#     fig, ax = plt.subplots()
#     n_gb = len(gbs)
#     gb_centroids = [gbs[i].centroid for i in range(n_gb)]
#     gb_centroids = np.array(gb_centroids)
#     # 绘制粒球的的质心
#     ax.scatter(gb_centroids[:, 0], gb_centroids[:, 1], c=gb_labels, cmap='plasma', marker='o', s=10)
#     ax.set_title("GB-DBSCAN Clustering Result (GB View)")
#     plt.show()
#
#
# def get_sample_labels(gb_labels: np.ndarray, gbs) -> np.ndarray:
#     """获取数据集中每一个样本的簇标签"""
#     n_samples = 0
#     for gb in gbs:
#         # 计算数据集样本点个数，用于初始化sample_labels
#         n_samples += len(gb.indices)
#
#     # 初始化每一个样本的簇标签为-1
#     sample_labels = np.full(n_samples, -1)
#     for i, gb in enumerate(gbs):
#         for sample_idx in gb.indices:
#             # 样本的簇标签和它所属的粒球的簇标签保持一致
#             sample_labels[sample_idx] = gb_labels[i]
#
#     return sample_labels
#
#
# def visualize_gb_dbscan_result_samples(dataset: np.ndarray, sample_labels: np.ndarray) -> None:
#     """可视化GB-DBSCAN聚类结果(样本点视角)"""
#     plt.scatter(dataset[:, 0], dataset[:, 1], c=sample_labels, cmap='plasma', s=10, marker='o')
#     plt.title("DBSCAN Clustering Result")
#     plt.show()
#
#
# def main() -> None:
#     """
#     1. 生成粒球空间
#     2. 对粒球空间进行DBSCAN聚类，同时得到每一个粒球的簇标签
#     3. 获取每一个样本点的簇标签，然后可视化GB-DBSCAN聚类结果
#     """
#     dataset = np.loadtxt('sample.txt')  # 加载数据集
#     gbs = generate_gbs(dataset)  # 生成粒球空间
#     verify_gbs(gbs)  # 验证粒球空间的有效性
#     visualize_original_data(dataset)  # 可视化原始数据
#     visualize_gbs(gbs)  # 可视化粒球空间
#
#     # 对粒球空间进行DBSCAN聚类，然后获取每一个粒球的簇标签
#     gb_centroids = np.array([gbs[i].centroid for i in range(len(gbs))])
#     eps, min_samples = 0.75, 6
#     clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(gb_centroids)
#     gb_labels = clustering.labels_
#     print(np.unique(gb_labels))
#
#     # 可视化GB-DBSCAN聚类结果(粒球视角)
#     visualize_gb_dbscan_result_gbs(gbs, gb_labels)
#
#     # 基于粒球簇标签，获取样本点的簇标签
#     sample_labels = get_sample_labels(gb_labels, gbs)
#     print(np.unique(sample_labels))
#
#     # 可视化GB-DBSCAN聚类结果(样本点视角)
#     visualize_gb_dbscan_result_samples(dataset, sample_labels)
#
#
# if __name__ == '__main__':
#     main()
