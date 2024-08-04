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
    n_samples = data.shape[0]
    np.random.seed(100)

    # 判断data, k, max_iterations是否合法
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array")
    if k <= 0 or k > n_samples or max_iterations < 1:
        raise ValueError("Invalid number of clusters")

    # 初始化k个聚类中心和样本的簇标签
    centroids = data[np.random.choice(n_samples, k, replace=False)]
    # # 初始化clusters，无物理含义，只是为了符合函数返回值的数据类型
    # clusters = [np.zeros(k) for _ in range(k)]  # for循环迭代中要让clusters指向空列表
    n_iteration = 0
    while True:
        n_iteration += 1
        # distances是一个形状为(k, n_samples)的np.ndarray实例，表示每一个聚类中心和其他所有点的距离
        distances = np.sqrt(np.sum(np.square(data-centroids[:, np.newaxis, :]), axis=2))

        # 为了和其他聚类算法保持一致和计算聚类算法的有效性指标，让函数返回clusters而不是labels
        labels = np.argmin(distances, axis=0)
        clusters = list()
        for i in range(k):
            clusters.append(np.where(labels == i)[0])

        # 更新聚类中心
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # 检查算法是否收敛(收敛的条件有很多, 不唯一, 选择一种方法即可)
        if np.all(np.abs(new_centroids - centroids) < tolerance) or n_iteration > max_iterations:
            break  # 新聚类中心和旧聚类中心相比变化很小，则停止迭代，聚类完成
        centroids = new_centroids

    # centroids, ndim=2, shape=(k, m_features), 表示k个聚类中心
    # labels, ndim=1, shape=(n_samples), 表示每一个数据点的簇标签
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
    np.random.seed(100)

    # 判断data, k, max_iterations是否合法
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array")
    if k <= 0 or k > n_samples or max_iterations < 1:
        raise ValueError("Invalid number of clusters")

    # 初始化k个聚类中心
    centroids = data[np.random.choice(n_samples, k, replace=False)]
    # 开始迭代
    n_iteration = 0
    while True:
        n_iteration += 1
        # distances, np.ndarray, shape=(k, n_samples); 表示k个聚类中心到n_samples个数据点的距离
        distances = np.sqrt(np.sum(np.square(data-centroids[:, np.newaxis, :]), axis=2))
        # judge: np.ndarray, shape=(k, n_samples)
        judge = distances - np.min(distances, axis=0)
        # bool_judge: np.ndarray, shape=(k, n_samples)
        bool_judge = judge < epsilon
        # 计算每一个簇包含的数据点
        clusters = list()
        for i in range(k):
            # 每一个cluster包含属于这个簇的所有数据点(实际上是数据点在数据集中的索引)
            cluster = np.where(bool_judge[i, :] == 1)[0]
            clusters.append(cluster)

        # 计算新的聚类中心
        new_centroids = list()
        for i in range(k):
            new_centroids.append(np.mean(data[clusters[i]], axis=0))
        new_centroids = np.array(new_centroids)

        # 检查算法是否收敛(收敛的条件有很多, 不唯一, 选择一种方法即可)
        if np.all(np.abs(new_centroids - centroids) < tolerance) or n_iteration > max_iterations:
            break  # 新聚类中心和旧聚类中心相比变化很小，则停止迭代，聚类完成

        # 若算法未收敛，则更新聚类中心，进行下一次迭代
        centroids = new_centroids

    # centroids, ndim=2, shape=(k, m_features), 表示k个聚类中心
    # clusters: list[np.ndarray], len(clusters)=k, 每个列表元素存储属于该簇的样本
    return centroids, clusters


def get_cores_fringes(clusters: list[np.ndarray]) -> tuple[list[set], list[set], list[set]]:
    """
    基于3WK-Means返回的clusters，求出每一个簇的支集，核心域集和边缘域集
    :param clusters: list[np.ndarray]，每一个列表元素包含一个簇的支集
    :return: tuple[list[set], list[set], list[set]]
        - 返回每一个簇的支集、核心域集和边缘域集
        - 对于任意一个簇，支集 = 核心域集合 + 边缘域集合
        - 即任意一个簇C_i, clusters[i] = cores[i] union fringes[i]
    """
    k = len(clusters)
    clusters = [set(clusters[i]) for i in range(k)]  # clusters[np.ndarray] -> clusters[set]
    cores = [set() for _ in range(k)]
    fringes = [set() for _ in range(k)]
    for i in range(k):
        m, n = (i+1) % k, (i+2) % k
        # 簇i核心域的样本点只能在i中，不能在簇(i+1)%k中，也不能在簇(i+2)%k中
        cores[i] = (clusters[i].difference(clusters[m])).intersection(clusters[i].difference(clusters[n]))
        # cores[i]和fringes[i]的并集就是cores[i]，也就是簇i的支集
        fringes[i] = clusters[i].difference(cores[i])

    # clusters, cores, fringes都是包含k个元素的列表，列表元素都是集合
    return clusters, cores, fringes


def visualize_original_data(data: np.ndarray, ax: plt.Axes) -> None:
    """
    可视化原始数据
    :param data: np.ndarray, shape=(n_samples, m_features), ndim=2
    :param ax: an instance of plt.Axes
    :return: None
    """
    # 在聚类之前可视化数据分布
    ax.scatter(data[:, 0], data[:, 1], s=5, marker='.', color='black')
    ax.set_title('Original Data')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_aspect('equal', adjustable='box')


def visualize_kmeans_results(data: np.ndarray, centroids: np.ndarray,
                             clusters: list[np.ndarray], ax: plt.Axes) -> None:
    """
    K-Means聚类结果可视化
    :param data: np.ndarray, shape=(n_samples, m_features), ndim=2
    :param centroids: np.ndarray, shape=(k, m_features), ndim=1
    :param clusters: np.ndarray, shape=(n_samples), the cluster label for each sample
    :param ax: an instance of plt.Axes
    :return: None
    """
    # 根据clusters计算所有样本的簇标签
    k = len(clusters)
    labels = [-1 for _ in range(len(data))]
    for i in range(k):
        for sample_idx in clusters[i]:
            labels[sample_idx] = i
    labels = np.array(labels)

    # 绘制数据点
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=5, marker='.')

    # 绘制聚类中心
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, marker='x')

    # 设置图片属性和样式
    ax.set_title('K-Means Clustering')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_aspect('equal', adjustable='box')


def visualize_twkmeans_results(data: np.ndarray, centers: np.ndarray,
                               clusters: list[np.ndarray], ax: plt.Axes) -> None:
    """
    3WK-Means聚类结果可视化
    :param data: np.ndarray, shape=(n_samples, m_features), ndim=2
    :param centers: np.ndarray, shape=(k, m_features), ndim=1
    :param clusters: list[np.ndarray], len(clusters)=k
    :param ax:
    :return:
    """
    # 三支聚类结果可视化
    k = len(clusters)
    # 获取每一个簇的支集，核心域集，边缘域集
    clusters, cores, fringes = get_cores_fringes(clusters)
    # 给核心域的数据点打上簇标签
    labels = list()
    for i in range(k):
        label = [i for _ in cores[i]]
        labels.append(label)

    # colors列表的长度要和k保持一致
    colors = ['yellow', 'green', 'blue']
    # 绘制核心域的点
    for i in range(k):
        # core_i_data表示簇i的核心域的数据点
        core_i_data = data[list(cores[i])]  # 集合不可以做np.ndarray的索引，列表可以
        ax.scatter(core_i_data[:, 0], core_i_data[:, 1], c=colors[i], marker='.', s=5)

    # 绘制边缘域的点
    fringes_indices = (fringes[0]).union(fringes[1]).union(fringes[2])
    # a = a.union(fringes[1])
    # a = a.union(fringes[2])
    fringe_data = data[list(fringes_indices)]
    ax.scatter(fringe_data[:, 0], fringe_data[:, 1], c='black', s=5, marker='.')

    # 绘制聚类中心
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=50, marker='x')

    # 设置图表标题，x轴标题，y轴标题
    ax.set_title('3WK-Means Clustering')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
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
