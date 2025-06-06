import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from visualize_original_data import visualize_original_data


def three_way_kmeans(data: np.ndarray, k: int = 3, max_iterations: int = 100, tolerance: float = 1e-4,
                     epsilon: float = 2.64) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    三支K-Means，每一个簇由核心域和边缘域两个集合表示
    原则：核心域的数据点只能属于一个簇的核心域，但是边缘域的数据点可以属于一个或者多个簇的边缘域
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
        distances = pairwise_distances(centroids, data)

        # judge: shape=(k, n_samples)
        judge = distances - np.min(distances, axis=0)
        # bool_judge: shape=(k, n_samples)
        bool_judge = (judge < epsilon)

        # 存储每一个簇包含的数据点在原始数据集中的索引
        clusters = list()
        for i in range(k):
            # cluster存储属于簇i的样本在原始数据集中的索引
            cluster = np.where(bool_judge[i, :] == 1)[0]
            clusters.append(cluster)

        # 计算新的聚类中心
        new_centroids = np.array([np.mean(data[clusters[i]], axis=0) for i in range(k)])

        # 检查算法是否收敛
        if np.all(np.abs(new_centroids - centroids) < tolerance) or n_iteration > max_iterations:
            # 新聚类中心和旧聚类中心相比变化很小或者达到最大迭代次数, 则停止迭代
            break

        # 若算法未收敛, 则更新聚类中心, 进行下一次迭代
        centroids = new_centroids

    # centroids, shape=(k, m_features), 表示k个聚类中心; clusters: list[np.ndarray], len(clusters)=k
    return centroids, clusters


def get_coredata_corelabels(data: np.ndarray, clusters: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    TODO 基于clusters和data，获取每一个簇的核心域样本点的簇标签和每一个簇的样本点集合
    1. 对于K-Means，返回一个data的复制品和每一个样本的簇标签(K-Means产生的簇就是核心域)
    2. 对于3WK-Means，返回属于核心域样本集合cores_data和核心域样本的簇标签cores_labels
    3. 此函数是为了计算聚类算法的validity index而设计的
    """
    k = len(clusters)
    cores, _ = get_cores_fringes(clusters)

    # 计算样本总数以初始化labels
    n_samples = len(data)

    # 初始化每一个数据集每一个样本的簇标签
    labels = np.full(n_samples, -1)

    # 修改labels: 核心域的样本分配相应的簇标签, 边缘域的样本簇标签保持为-1
    for i in range(k):
        for sample_idx in cores[i]:
            labels[sample_idx] = i

    # indices是核心域的样本点在原始数据集中的索引
    indices = np.where(labels != -1)[0]

    # 获取核心域样本的簇标签
    cores_labels = labels[indices]

    # 获取核心域的样本点集合
    cores_data = data[indices]

    return cores_data, cores_labels


def get_cores_fringes(clusters: list[np.ndarray]) -> tuple[list[np.array], list[np.array]]:
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

    # 计算每一个簇的核心域和边缘域
    for i in range(k):
        cores[i] = clusters[i]
        for j in range(k):
            if i != j:
                # 簇i核心域的样本点只能在i中，不能在任何其它簇中出现
                cores[i] = cores[i].intersection(clusters[i].difference(clusters[j]))

        # cores[i]和fringes[i]的并集就是clusters[i]，clusters[i]亦称为簇i的支集
        fringes[i] = clusters[i].difference(cores[i])

    # 列表中的元素由集合转换为np.ndarray
    cores = [np.array(list(core)) for core in cores]
    fringes = [np.array(list(fringe)) for fringe in fringes]

    # cores是包含k个元素的列表, 每个元素是一维np.ndarray, fringes同理
    return cores, fringes


def visualize_twkmeans_result(data: np.ndarray, centers: np.ndarray,
                              clusters: list[np.ndarray]) -> None:
    """
    可视化3WK-Means聚类结果
    :param data: np.ndarray, shape=(n_samples, m_features), ndim=2
    :param centers: np.ndarray, shape=(k, m_features), ndim=1
    :param clusters: list[np.ndarray], len(clusters)=k
    :return:
    """
    fig, ax = plt.subplots()
    # 簇的个数k和数据集中的样本数量n_samples
    k = len(clusters)
    n_samples = len(data)

    # cores是包含k个元素的列表, 每个元素是一维np.ndarray数组, fringes同理
    cores, fringes = get_cores_fringes(clusters)

    # 为每一个样本分配簇标签(边缘域样本的簇标签保持为-1)
    labels = np.full(n_samples, -1)
    for i in range(k):
        for sample_idx in cores[i]:
            labels[sample_idx] = i

    # 绘制数据点
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='plasma', s=5, marker='.')
    # 绘制聚类中心
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=50, marker='x')

    # 为了凸显边缘域的样本点，覆盖为黑色
    fringes_data = data[np.unique(np.concatenate(fringes))]
    ax.scatter(fringes_data[:, 0], fringes_data[:, 1], c='black', s=5, marker='.')

    # 设置图片属性和样式
    ax.set_title('3WK-Means Clustering')
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def main() -> None:
    """
    1. 数据集sample.txt, (4000, 2)
    2. 3WK-Means, k=3, epsilon用于判断一个数据点是否属于一个簇的支集, epsilon越大, 边缘域的点越多
    3. 可视化3W-KMeans聚类结果
    """

    # 载入数据
    dataset = np.loadtxt('sample.txt')

    # 获取KMeans聚类结果
    centroids, clusters = three_way_kmeans(dataset, k=3, epsilon=2.64)

    # 可视化原始数据
    visualize_original_data(dataset)

    # 可视化3W-KMeans聚类结果
    visualize_twkmeans_result(dataset, centroids, clusters)


if __name__ == '__main__':
    main()
