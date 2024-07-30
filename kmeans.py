"""
熟悉numpy中常用函数和numpy的广播机制

1. np.random.choice, np.argmin, np.all, np.any, np.full
2. 关于arr的shape=(2, 3, 4)
    理解为：共两页，每页三行，每行四列
    也即：共2个二维数组，每个二维数组有3个一维数组，每个一维数组有4个元素
3. np的广播机制的基本原则：首先看维度数是否相同！！！
    - 如果两个数组的维度数不同，形状较小的数组会在前面补充维度为1
    - 若两个数组的维度数相同，且两个数组的形状在某个维度上不匹配，但其中一个的长度为1，则该维度会被扩展以匹配另一个数组
    - 若两个数组的维度数相同，且两个数组的形状在某个维度上不匹配，但两者都不为1，则会报错
    如
4. np的增维原则
    假设a = [[4, 9, 9, 4], [5, 6, 6, 7]]), a的形状是(2, 4)
    则b = a[:, np.newaxis, :]的形状就是(2, 1, 4)
    则b =[[[4, 9, 9, 4]], [[5, 6, 6, 7]]]
"""


import numpy as np
import matplotlib.pyplot as plt


def kmeans(data: np.ndarray, k: int = 3, max_iterations: int = 100, tolerance: float = 1e-4):
    """
    K-Means算法实现

    :param data: np.ndarray, shape=(n_samples, n_features), ndim=2
    :param k: number of clusters
    :param max_iterations: the number of maximum iterations
    :param tolerance: float, tolerance for convergence
    :return: tuple[np.ndarray, np.ndarray]
        - centroids: np.ndarray, shape=(k, n_features), cluster centers
        - labels: np.ndarray, shape=(n_samples,), cluster labels for each sample
        - n_iteration: int, number of iterations of K-Means algorithm
    """
    n_samples = data.shape[0]
    np.random.seed(n_samples)
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        # 判断data是否有效
        raise ValueError("Data must be a 2D numpy array")
    if k <= 0 or k > n_samples:
        # 判断k是否有效
        raise ValueError("Invalid number of clusters")
    centroids = data[np.random.choice(n_samples, k, replace=False)]  # 初始化聚类中心
    labels = np.full(n_samples, -1)  # 样本的簇标签初始化为-1
    n_iteration = 0
    while n_iteration < max_iterations:
        # distances是一个形状为(k, n_samples)的数组，表示每一个聚类中心和其他所有点的距离，理解numpy广播机制
        distances = np.sqrt(np.sum(np.square(data-centroids[:, np.newaxis, :]), axis=2))
        labels = np.argmin(distances, axis=0)
        # 更新聚类中心
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        # 检查算法是否收敛(收敛的条件有很多, 不唯一, 选择一种方法即可)
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break  # 新聚类中心和旧聚类中心相比变化很小，则停止迭代，聚类完成
        centroids = new_centroids
        n_iteration += 1

    return centroids, labels, n_iteration


def visualization_before_clustering(data, ax):
    # 在聚类之前可视化数据分布
    ax.scatter(data[:, 0], data[:, 1])
    ax.set_title('Visualization Before Clustering')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')


def visualization_after_clustering(data, centroids, labels, ax):
    # 聚类结果可视化
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='*')
    ax.set_title('Visualization After Clustering')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')


def main():
    # 生成数据
    data = np.loadtxt('sample.txt')
    # K-Means聚类算法
    centroids, labels, n_iteration = kmeans(data, k=5, max_iterations=1000, tolerance=1e-4)
    print(f"Number of Iterations: {n_iteration}")
    # 数据可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    visualization_before_clustering(data, ax1)
    visualization_after_clustering(data, centroids, labels, ax2)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
