"""
熟悉numpy中常用函数

1. np.random.choice
2. np.argmin
3. np.all
4. np的广播机制的基本原则
    - 如果两个数组的维度数不同，形状较小的数组会在前面补充维度为1
    - 如果两个数组的形状在某个维度上不匹配，但其中一个的长度为1，则该维度会被扩展以匹配另一个数组
    - 如果两个数组的形状在某个维度上不匹配，且两者都不为1，则会报错
"""


import numpy as np
import matplotlib.pyplot as plt


def kmeans(data, k=4, max_iters=100):
    """
    k-means算法实现

    :param data: 数据集，numpy数组
    :param k: 簇个数，int
    :param max_iters: 最大迭代次数，int

    :return: 返回一个包含两个元素的元组
        - 聚类中心centroids，numpy数组
        - 样本标签labels，numpy数组，表示每一个样本所属的簇
    """
    # 随机初始化k个聚类中心, centroid表示形心，质心或者重心
    # np.random.choice(n_samples, k, replace=False): 从[0, 1, ... , 3999]中不放回的选择k个值
    n_samples = data.shape[0]
    np.random.seed(n_samples)
    centroids = data[np.random.choice(n_samples, k, replace=False)]
    labels = []
    for _ in range(max_iters):
        # 将每一个对象分配给最近的聚类中心
        # distances是一个形状为(k, n_samples)的数组，表示每一个聚类中心和其他所有点的距离，理解numpy广播机制
        distances = np.sqrt(np.sum(np.square(data-centroids[:, np.newaxis]), axis=2))
        # 每一个数据点都会找到自己所属的聚类中心，并将聚类中心的编号(0, 1, ... , k-1)作为数据点的簇标签
        # labels是一个形状为(n_samples,)的numpy数组，保存了每一个点的簇标签
        labels = np.argmin(distances, axis=0)
        # 更新聚类中心；基于刚刚得到的簇，计算每一个簇新的聚类中心
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        # 检查算法是否收敛，收敛的条件有很多，并不唯一，可以选择一中方法即可
        if np.all(centroids == new_centroids):
            # 新的聚类中心和旧的聚类中心相比没有变化，则停止迭代，聚类完成
            break
        centroids = new_centroids
    return centroids, labels


def visualization_after_clustering(data, centroids, labels, ax):
    # 聚类结果可视化
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='*')
    ax.set_title('Visualization After Clustering')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')


def visualization_before_clustering(data, ax):
    # 在聚类之前可视化数据分布
    ax.scatter(data[:, 0], data[:, 1])
    ax.set_title('Visualization Before Clustering')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')


def main():
    # 生成数据
    data = np.loadtxt('sample.txt')
    centroids, labels = kmeans(data, 5, 100)
    # 数据可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    visualization_before_clustering(data, ax1)
    visualization_after_clustering(data, centroids, labels, ax2)
    # 紧凑布局：tight_layout自动调整子图的位置和大小，以确保所有的标签、标题等元素都能完整显示，同时最大化图形区域的使用
    fig.tight_layout()
    # 显示图片
    plt.show()


if __name__ == '__main__':
    main()
