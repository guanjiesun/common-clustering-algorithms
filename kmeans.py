import numpy as np
import matplotlib.pyplot as plt


def kmeans(data: np.ndarray, k: int = 3, max_iterations: int = 100,
           tolerance: float = 1e-4) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    K-Means算法实现(将kmeans升级为kmeans++)
    :param data: np.ndarray, shape=(n_samples, n_features), ndim=2
    :param k: number of clusters
    :param max_iterations: the number of maximum iterations
    :param tolerance: float, tolerance for convergence
    :return: tuple[np.ndarray, np.ndarray]
        - centroids: np.ndarray, shape=(k, n_features), cluster centers
        - labels: np.ndarray, shape=(n_samples), the cluster label for each sample
    """
    n_samples = len(data)

    # 判断data, k, max_iterations是否合法
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError("Data must be a 2D numpy array!")
    if k <= 0 or k > n_samples:
        raise ValueError("Invalid number of clusters!")
    if max_iterations < 1:
        raise ValueError("Invalid max_iterations!")

    # 初始化k个聚类中心和样本的簇标签
    np.random.seed(100)
    centroids = data[np.random.choice(n_samples, k, replace=False)]

    n_iteration = 0
    while True:
        n_iteration += 1
        # distances, shape=(k, n_samples); 表示k个聚类中心到n_samples个数据点的距离
        distances = np.sqrt(np.sum(np.square(data-centroids[:, np.newaxis, :]), axis=2))

        # 函数返回clusters而不是labels
        labels = np.argmin(distances, axis=0)
        clusters = list()
        for i in range(k):
            # 每一个cluster存储属于簇i的所有样本(实际上是样本在原始数据集中的索引)
            clusters.append(np.where(labels == i)[0])

        # 计算新的聚类中心
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # 检查算法是否收敛(新聚类中心和旧聚类中心相比变化很小或者达到最大迭代次数, 则停止迭代)
        if np.all(np.abs(new_centroids - centroids) < tolerance) or n_iteration > max_iterations:
            break  # 新聚类中心和旧聚类中心相比变化很小，则停止迭代，聚类完成
        centroids = new_centroids

    # centroids, shape=(k, m_features), 表示k个聚类中心
    # clusters: list[np.ndarray], len(clusters)=k
    return centroids, clusters


def three_way_kmeans(data: np.ndarray, k: int = 3, max_iterations: int = 100, tolerance: float = 1e-4,
                     epsilon: float = 2.64) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    三支K-Means，每一个簇有核心域和边缘域两个集合表示
    原则：每一个数据点只能属于一个簇的核心域，但是可以属于一个或者多个簇的边缘域
    论文：于洪《三支聚类综述》；王平心《三支K-Means》
    :param data: np.ndarray, shape=(n_samples, m_features), ndim=2
    :param k: int, to specify the number of cluster
    :param max_iterations: int, maximum iterations limited to memory and cpu
    :param tolerance: termination condition of iteration
    :param epsilon: the hyperparameter of 3WK-Means, inspired by 王平心's 3WK-Means
    :return: tuple[np.ndarray, list[np.ndarray]]
        - centroids: np.ndarray, shape=(k, m_features), ndim=2
        - clusters: list[np.ndarray], include k np.ndarray
    """
    n_samples = len(data)

    # 判断data, k, max_iterations是否合法
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array")
    if k <= 0 or k > n_samples or max_iterations < 1:
        raise ValueError("Invalid number of clusters")

    # 初始化k个聚类中心
    np.random.seed(100)
    centroids = data[np.random.choice(n_samples, k, replace=False)]

    # 开始迭代
    n_iteration = 0
    while True:
        n_iteration += 1
        # distances, shape=(k, n_samples); 表示k个聚类中心到n_samples个数据点的距离
        distances = np.sqrt(np.sum(np.square(data-centroids[:, np.newaxis, :]), axis=2))
        # judge: shape=(k, n_samples)
        judge = distances - np.min(distances, axis=0)
        # bool_judge: shape=(k, n_samples)
        bool_judge = judge < epsilon
        # 计算每一个簇包含的数据点
        clusters = list()
        for i in range(k):
            # 每一个cluster存储属于簇i的所有样本(实际上是样本在原始数据集中的索引)
            cluster = np.where(bool_judge[i, :] == 1)[0]
            clusters.append(cluster)

        # 计算新的聚类中心
        new_centroids = list()
        for i in range(k):
            new_centroids.append(np.mean(data[clusters[i]], axis=0))
        new_centroids = np.array(new_centroids)

        # 检查算法是否收敛(新聚类中心和旧聚类中心相比变化很小或者达到最大迭代次数, 则停止迭代)
        if np.all(np.abs(new_centroids - centroids) < tolerance) or n_iteration > max_iterations:
            break

        # 若算法未收敛, 则更新聚类中心, 进行下一次迭代
        centroids = new_centroids

    # centroids, shape=(k, m_features), 表示k个聚类中心
    # clusters: list[np.ndarray], len(clusters)=k
    return centroids, clusters


def get_cores_fringes(clusters: list[np.ndarray]) -> tuple[list[np.array], list[np.array], list[np.array]]:
    """
    基于3WK-Means返回的clusters，求出每一个簇的支集，核心域集和边缘域集
    :param clusters: list[np.ndarray]，每一个列表元素包含一个簇的支集
    :return: tuple[list[set], list[set], list[set]]
        - 返回每一个簇的支集、核心域集和边缘域集
        - 对于任意一个簇，支集 = 核心域集合 + 边缘域集合
        - 即任意一个簇C_i, clusters[i] = cores[i] union fringes[i]
    """
    # TODO 如果clusters是K-Means产生的，也会返回正确的结果（边界域为空集）；3WK-Means就是K-Means的泛化模型！
    k = len(clusters)
    # np.ndarray转换为集合，方便运算
    clusters = [set(clusters[i]) for i in range(k)]  # clusters[np.ndarray] -> clusters[set]
    cores = [set() for _ in range(k)]
    fringes = [set() for _ in range(k)]
    for i in range(k):
        # cores[i]的初始化很重要，需要熟悉集合的运算
        cores[i] = clusters[i]
        for j in range(k):
            if i != j:
                cores[i] = cores[i].intersection(clusters[i].difference(clusters[j]))

        fringes[i] = clusters[i].difference(cores[i])

    # 列表中的元素由集合转换为np.ndarray
    clusters = [np.array(list(cluster)) for cluster in clusters]
    cores = [np.array(list(core)) for core in cores]
    fringes = [np.array(list(fringe)) for fringe in fringes]
    # clusters是包含k个元素的列表, 每个元素是一维np.ndarray数组; cores和fringes同理
    return clusters, cores, fringes


def visualize_original_data(data: np.ndarray, ax: plt.Axes) -> None:
    """
    可视化原始数据
    :param data: np.ndarray, shape=(n_samples, m_features), ndim=2
    :param ax: an instance of plt.Axes
    :return: None
    """
    ax.scatter(data[:, 0], data[:, 1], s=5, marker='.', color='black')
    ax.set_title('Original Data')
    ax.set_aspect('equal', adjustable='box')


def visualize_kmeans_results(data: np.ndarray, centroids: np.ndarray,
                             clusters: list[np.ndarray], ax: plt.Axes) -> None:
    """
    可视化K-Means聚类结果
    :param data: np.ndarray, shape=(n_samples, m_features), ndim=2
    :param centroids: np.ndarray, shape=(k, m_features), ndim=1
    :param clusters: np.ndarray, shape=(n_samples), the cluster label for each sample
    :param ax: an instance of plt.Axes
    :return: None
    """
    # 簇的个数k和数据集中的样本数量n_samples
    k = len(clusters)
    n_samples = len(data)

    # 为每一个样本分配簇标签
    labels = np.full(n_samples, -1)
    for i in range(k):
        for sample_idx in clusters[i]:
            labels[sample_idx] = i
    labels = np.array(labels)

    # 绘制数据点
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='plasma', s=5, marker='.')
    # 绘制聚类中心
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, marker='x')
    # 设置图片属性和样式
    ax.set_title('K-Means Clustering')
    ax.set_aspect('equal', adjustable='box')


def visualize_twkmeans_results(data: np.ndarray, centers: np.ndarray,
                               clusters: list[np.ndarray], ax: plt.Axes) -> None:
    """
    可视化3WK-Means聚类结果
    :param data: np.ndarray, shape=(n_samples, m_features), ndim=2
    :param centers: np.ndarray, shape=(k, m_features), ndim=1
    :param clusters: list[np.ndarray], len(clusters)=k
    :param ax:
    :return:
    """
    # 簇的个数k和数据集中的样本数量n_samples
    k = len(clusters)
    n_samples = len(data)

    # clusters是包含k个元素的列表, 每个元素是一维np.ndarray数组; cores和fringes同理
    clusters, cores, fringes = get_cores_fringes(clusters)

    # 为每一个样本分配簇标签(边缘域样本的簇标签保持为-1)
    labels = np.full(n_samples, -1)
    for i in range(k):
        for sample_idx in cores[i]:
            labels[sample_idx] = i

    # 绘制数据点
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='plasma', s=5, marker='.')
    # 绘制聚类中心
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=50, marker='x')

    # 为了凸显边缘域的数据点，覆盖为黑色
    fringes_data = data[np.unique(np.concatenate(fringes))]
    ax.scatter(fringes_data[:, 0], fringes_data[:, 1], c='black', s=5, marker='.')

    # 设置图片属性和样式
    ax.set_title('3WK-Means Clustering')
    ax.set_aspect('equal', adjustable='box')


def main() -> None:
    """
    样本数据集sample.txt, (4000, 2)
    1. K-Means, k=3
    2. 3WK-Means, k=3, epsilon判断一个数据点是否属于一个簇的支集, epsilon越大, 边缘域的点越多
    3. 可视化K-Means和3WK-Means的聚类结果
    :return: None
    """
    # 生成数据
    data = np.loadtxt('sample.txt')

    # 获取K-Means聚类结果
    centroids1, clusters1 = kmeans(data, k=3)
    # 获取3WK-Means聚类结果
    centroids2, clusters2 = three_way_kmeans(data, k=3, epsilon=2.64)

    # 数据可视化
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))
    visualize_original_data(data, ax0)
    visualize_kmeans_results(data, centroids1, clusters1, ax1)
    visualize_twkmeans_results(data, centroids2, clusters2, ax2)

    # 图片布局设置
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
