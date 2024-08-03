from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# TODO import data, ndim=2, shape=(n_samples, m_features)
# TODO data is a global variable
data = np.loadtxt('sample.txt')


class GranularBall:
    def __init__(self, indices):
        self.indices = indices

    def get_gb_data(self):
        return data[self.indices]

    def get_centroid(self) -> np.ndarray:
        return np.mean(self.get_gb_data(), axis=0)

    def get_radius(self) -> np.float64:
        return np.max(np.sqrt(np.sum(np.square(self.get_gb_data() - self.get_centroid()), axis=1)))


def kmeans(indices: np.ndarray, k=2, max_iterations=100, tolerance=1e-4):
    gb_data = data[indices]
    gb_size = len(gb_data)
    np.random.seed(gb_size)

    # 初始化k个聚类中心
    centroids = gb_data[np.random.choice(gb_size, k, replace=False)]
    n_iteration = 0
    while True:
        n_iteration += 1
        # distances: np.ndarray, shape=(k, gb_size), 表示每一个聚类中心和其他所有点的距离
        distances = np.sqrt(np.sum(np.square(gb_data-centroids[:, np.newaxis, :]), axis=2))
        # 计算每一个样本的簇标签
        labels = np.argmin(distances, axis=0)
        clusters = list()
        for i in range(k):
            # indices的作用：确保clusters中的数据点的索引是在原始数据集中的索引
            clusters.append(indices[np.where(labels == i)[0]])

        # 更新聚类中心
        new_centroids = np.array([gb_data[labels == i].mean(axis=0) for i in range(k)])

        # 检查算法是否收敛(收敛的条件有很多, 不唯一, 选择一种方法即可)
        if np.all(np.abs(new_centroids-centroids) < tolerance) or n_iteration > max_iterations:
            # 新聚类中心和旧聚类中心相比变化很小，或者达到最大跌打次数，则停止迭代，聚类完成
            break

        centroids = new_centroids

    return clusters


def split_granular_ball(gb: GranularBall) -> (GranularBall, GranularBall):
    # gb.indices表示在
    clusters = kmeans(gb.indices)
    micro_gb_1 = GranularBall(clusters[0])
    micro_gb_2 = GranularBall(clusters[1])

    return micro_gb_1, micro_gb_2


def generate_granular_balls() -> list[GranularBall]:
    gbs = list()
    # size表示原始数据集data的样本数量
    size = len(data)
    # 判断粒球是否需要进一步划分为两个小粒球的阈值
    threshold = np.sqrt(size)

    # 初始化队列；deque函数需要的参数是一个可迭代对象
    queue = deque([GranularBall(np.arange(size))])
    while queue:
        gb = queue.popleft()  # 使用popleft而不是pop是因为要保持queue先进先出的特性
        if len(gb.indices) > threshold:
            # 如果gb太大，则使用2means算法将gb划分为两个更小的粒球，然后两个小粒球入队
            queue.extend(split_granular_ball(gb))
        else:
            gbs.append(gb)

    return gbs


def verify_gbs(gbs: list[GranularBall]) -> None:
    # 使用Python中集合的特性，验证生成的粒球空间的正确性
    size = len(gbs)

    # 两两(粒球的数据索引)互不相交，则生成的粒球空间正确
    for i in range(size):
        set_i = set(gbs[i].indices)
        for j in range(i+1, size):
            set_j = set(gbs[j].indices)
            if set_i.intersection(set_j) != set():
                raise ValueError("Wrong Granular Ball Space")


def visualize_gbs(gbs, ax: plt.Axes):
    for i, gb in enumerate(gbs):
        # 获取粒球的数据、质心和半径
        gb_data, centroid, radius = data[gb.indices], gb.get_centroid(), gb.get_radius()

        # float_x, float_y和float_radius是为了符合Circle函数对参数数据类型的要求
        float_x, float_y = float(centroid[0]), float(centroid[1])
        float_radius = float(radius)

        # 创建圆, 绘制数据点, 绘制圆, 绘制圆心
        circle = Circle((float_x, float_y), float_radius, fill=False, color='blue')
        ax.scatter(gb_data[:, 0], gb_data[:, 1], color='black', marker='.', s=5)
        ax.add_artist(circle)
        ax.scatter(centroid[0], centroid[1], color='red', s=5)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def main():
    # generate gbs
    gbs = generate_granular_balls()
    verify_gbs(gbs)
    # visualize gbs
    fig, ax = plt.subplots(figsize=(8, 6))
    visualize_gbs(gbs, ax)


if __name__ == '__main__':
    main()
