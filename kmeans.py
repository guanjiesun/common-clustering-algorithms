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
    """
    n_samples = data.shape[0]
    np.random.seed(100)

    if not isinstance(data, np.ndarray) or data.ndim != 2:  # 判断data是否有效
        raise ValueError("Data must be a 2D numpy array")
    if k <= 0 or k > n_samples:
        raise ValueError("Invalid number of clusters")  # 判断k是否有效

    centroids = data[np.random.choice(n_samples, k, replace=False)]  # 初始化k个聚类中心
    labels = np.full(n_samples, -1)  # 数据点的簇标签初始化为-1
    for _ in range(max_iterations):
        # distances是一个形状为(k, n_samples)的np.ndarray实例，表示每一个聚类中心和其他所有点的距离
        distances = np.sqrt(np.sum(np.square(data-centroids[:, np.newaxis, :]), axis=2))
        labels = np.argmin(distances, axis=0)
        # 更新聚类中心
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        # 检查算法是否收敛(收敛的条件有很多, 不唯一, 选择一种方法即可)
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break  # 新聚类中心和旧聚类中心相比变化很小，则停止迭代，聚类完成
        centroids = new_centroids

    return centroids, labels


def three_way_kmeans(data: np.ndarray, k: int = 3, max_iterations: int = 100, tolerance=1e-4, epsilon=2.0):
    n_samples = len(data)
    np.random.seed(100)
    centroids = data[np.random.choice(n_samples, k, replace=False)]  # 初始化k个聚类中心
    for _ in range(max_iterations):
        # distances是一个形状为(k, n_samples)的np.ndarray实例，表示每一个聚类中心和其他所有点的距离
        distances = np.sqrt(np.sum(np.square(data-centroids[:, np.newaxis, :]), axis=2))
        judge = distances - np.min(distances, axis=0)
        bool_judge = judge < epsilon
        # 计算每一个簇包含的数据点
        clusters = list()
        for i in range(k):
            cluster = np.where(bool_judge[i, :] == 1)[0]
            clusters.append(cluster)
        return centroids, clusters


def visualization_before_clustering(data, ax):
    # 在聚类之前可视化数据分布
    ax.scatter(data[:, 0], data[:, 1], s=5, marker='.', color='black')
    ax.set_title('Visualization Before Clustering')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')


def visualization_after_kmeans_clustering(data, centroids, labels, ax):
    # 聚类结果可视化
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=5, marker='.')
    ax.scatter(centroids[:, 0], centroids[:, 1], color='red', s=20, marker='.')
    ax.set_title('Visualization After K-Means Clustering')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')


def visualization_after_three_way_kmeans_clustering(data, centroids, clusters, ax):
    # 三支聚类结果可视化
    k = len(clusters)  # k保存簇的个数
    clusters = [set(clusters[i]) for i in range(k)]  # clusters[np.ndarray] -> clusters[set]
    cores = [set() for _ in range(k)]
    fringes = [set() for _ in range(k)]
    for i in range(k):
        m, n = (i+1) % k, (i+2) % k
        # 簇i核心域的样本点只能在i中，不能在簇(i+1)%k中，也不能在簇(i+2)%k中
        cores[i] = (clusters[i].difference(clusters[m])).intersection(clusters[i].difference(clusters[n]))
        # cores[i]和fringes[i]的并集就是cores[i]，也就是簇i的支集
        fringes[i] = clusters[i].difference(cores[i])

    # 给核心域的数据点打上簇标签
    labels = list()
    for i in range(k):
        label = [i for _ in cores[i]]
        labels.append(label)

    # 绘制核心域的点
    for i in range(k):
        colors = ['yellow', 'green', 'blue']
        core_i_data = data[list(cores[i])]  # 集合不可以做np.ndarray的索引，列表可以
        ax.scatter(core_i_data[:, 0], core_i_data[:, 1], c=colors[i], marker='.', s=5)

    # 绘制边缘域的点
    a = (fringes[0]).union(fringes[1])
    a = a.union(fringes[1])
    a = a.union(fringes[2])
    fringe_data = data[list(a)]
    ax.scatter(fringe_data[:, 0], fringe_data[:, 1], c='black', s=20, marker='.')

    # 绘制聚类中心
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=20, marker='.')

    # 设置图表标题，x轴标题，y轴标题
    ax.set_title('Visualization After 3WK-Means Clustering')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')


def main():
    # 生成数据
    data = np.loadtxt('sample.txt')

    # 获取K-Means聚类结果
    centroids, labels = kmeans(data, k=3, max_iterations=1000, tolerance=1e-4)
    # 获取3WK-Means聚类结果
    centers, clusters = three_way_kmeans(data, k=3, max_iterations=1000, tolerance=1e-4)

    # 数据可视化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    visualization_before_clustering(data, ax1)
    visualization_after_kmeans_clustering(data, centroids, labels, ax2)
    visualization_after_three_way_kmeans_clustering(data, centers, clusters, ax3)

    # 图片布局设置
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
