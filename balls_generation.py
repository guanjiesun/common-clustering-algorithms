from collections import deque  # Standard library imports

import numpy as np  # Third-party library imports
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from kmeans import kmeans  # Local module imports


class GranularBall:
    """
    A Granular Ball is a hypersphere in a multidimensional space that encapsulates
    a set of data points. It is characterized by its centroid and radius.

    Attributes:
        data (np.ndarray): The data points contained in the ball.
                           Shape: (n_samples, m_features)
                           Ndim: 2
        size (int): The number of data points in the ball.
        indices (np.ndarray): The index in origin dataset of each datapoint within Granular Ball
                              shape: (gb.size)
                              ndim: 1

    Methods:
        get_centroid(): Calculates and returns the centroid of the ball.
        get_radius(): Calculates and returns the radius of the ball.
    """

    def __init__(self, data: np.ndarray, indices: np.ndarray = None):
        """
        Initializes a GranularBall instance.

        Args:
            data (np.ndarray): The data points to be enclosed in the ball.
                               Shape: (n_samples, m_features)
                               Ndim: 2

        Raises:
            ValueError: If the input data is not a 2D numpy array.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Data must be a 2D numpy array")

        self.data = data
        self.size = len(data)

        if indices is None:
            self.indices = np.arange(len(data))  # 将整个数据集初始化为一个粒球的时候需要用到此行代码
        else:
            self.indices = indices  # 粒球中的数据点在原始数据集中的索引

    def get_centroid(self) -> np.ndarray:
        """
        Calculates the centroid of the Granular Ball.

        Returns:
            np.ndarray: The centroid of the ball. Shape: (m_features). Ndim: 1
        """
        return np.mean(self.data, axis=0)

    def get_radius(self) -> np.float64:
        """
        Calculates the radius of the Granular Ball.
        The radius is defined as the maximum distance from any point to the centroid of the ball.

        Returns:
            np.float64: The radius of the ball.
        """
        return np.max(np.sqrt(np.sum(np.square(self.data - self.get_centroid()), axis=1)))


def split_granular_ball(gb: GranularBall) -> (GranularBall, GranularBall):
    """
    split a gb into two small gbs using 2means clustering algorithm

    :param gb: Granular Ball
    :return: tuple(GranularBall, GranularBall)
    """
    _, labels = kmeans(gb.data, 2)
    # 获取满足条件的索引; np.where返回的只包含一个np.ndarray的元组
    indices_1 = np.where(labels == 0)[0]
    indices_2 = np.where(labels == 1)[0]
    # 使用np的高级索引创建视图(引用)而不是复制，复制会占用大量内存空间
    micro_gb_1 = GranularBall(gb.data[indices_1], indices=indices_1)
    micro_gb_2 = GranularBall(gb.data[indices_2], indices=indices_2)

    return micro_gb_1, micro_gb_2


def generate_granular_balls(data: np.ndarray) -> list[GranularBall]:
    """
    generate gbs based on a given dataset

    :param data: np.ndarray, shape=(n_samples, m_features)
    :return: gbs, a list of Granular Balls, abbreviated as gbs
    """
    gbs = list()  # 初始化粒球列表
    threshold = np.sqrt(len(data))  # 判断粒球是否需要进一步划分为两个小粒球的阈值
    queue = deque([GranularBall(data)])  # 初始化队列；deque函数需要的参数是一个可迭代对象
    while queue:
        gb = queue.popleft()  # 使用popleft而不是pop是因为要保持queue先进先出的特性
        if gb.size > threshold:
            # 如果gb太大，则使用2means算法将gb划分为两个更小的粒球，然后两个小粒球入队
            queue.extend(split_granular_ball(gb))
        else:
            gbs.append(gb)

    return gbs


def print_gb_indices(gbs: list[GranularBall]) -> None:
    """
    输出gbs中每一个粒球包含的对象在原始数据集中的索引信息

    :param gbs: a list of Granular Balls
    :return: None
    """
    for gb in gbs:
        print(gb.indices)


def plot_granular_balls(gbs: list[GranularBall], ax: plt.Axes) -> None:
    """
    visualize gbs(只能可视化具有两个特征的数据集，若数据集有三个或者以上的特征，需要重新设计函数)

    :param gbs: a list of Granular Balls
    :param ax: an instance of plt.Axes
    :return: None
    """
    for i, gb in enumerate(gbs):
        # 获取粒球的数据、质心和半径
        data, centroid, radius = gb.data, gb.get_centroid(), gb.get_radius()
        # float_x, float_y和float_radius是为了符合Circle函数对参数的要求
        float_x, float_y = float(centroid[0]), float(centroid[1])
        float_radius = float(radius)
        # 创建圆
        circle = Circle((float_x, float_y), float_radius, fill=False, color='blue')
        # 绘制数据点
        ax.scatter(data[:, 0], data[:, 1], color='blue', marker='.', s=5)
        # 绘制圆
        ax.add_artist(circle)
        # 绘制圆心
        ax.scatter(centroid[0], centroid[1], color='red', s=5)
        # 设置图片的纵横比和调整图形的方式
        ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    plt.show()


def main():
    """
    generate gbs and visualize gbs, gbs represents "Granular Balls" or "Granular Ball Space"

    :return: None
    """
    # import data, data.ndim=2, data.shape=(n_samples, m_features)
    data = np.loadtxt('sample.txt')
    # generate gbs
    gbs = generate_granular_balls(data)
    # visualize gbs
    fig, ax = plt.subplots()
    plot_granular_balls(gbs, ax)


if __name__ == '__main__':
    main()
