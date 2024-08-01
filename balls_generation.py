from collections import deque  # Standard library imports

import numpy as np  # Third-party library imports
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from kmeans import kmeans  # Local module imports


class GranularBall:
    """粒球"""
    def __init__(self, data: np.ndarray):
        """
        initiate a granular ball

        :param data: np.ndarray, ndim=2, shape=(n_samples, m_features)
        """
        self.data = data
        self.size = len(data)

    def get_centroid(self) -> np.ndarray:
        """
        get the centroid of one granular ball

        :return: np.ndarray
        """
        return np.mean(self.data, axis=0)

    def get_radius(self) -> np.float64:
        """
        get the radius of one granular ball

        :return: float
        """
        return np.max(np.sqrt(np.sum(np.square(self.data-self.get_centroid()), axis=1)))


def split_granular_ball(gb: GranularBall) -> (GranularBall, GranularBall):
    """
    split a gb into two small gbs using 2means clustering algorithm

    :param gb: Granular Ball
    :return: tuple(GranularBall, GranularBall)
    """
    _, labels = kmeans(gb.data, 2)
    micro_gb_1 = GranularBall(gb.data[labels == 0])
    micro_gb_2 = GranularBall(gb.data[labels == 1])

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


def plot_granular_balls(gbs: list[GranularBall], ax: plt.Axes) -> None:
    """
    visualize gbs

    :param gbs: a list of Granular Balls
    :param ax: an instance of plt.Axes
    :return: None
    """
    for i, gb in enumerate(gbs):
        # 获取粒球包含的数据、粒球形心和粒球半径
        data, centroid, radius = gb.data, gb.get_centroid(), gb.get_radius()
        float_x, float_y = float(centroid[0]), float(centroid[1])
        float_radius = float(radius)
        circle = Circle((float_x, float_y), float_radius, fill=False, color='blue')  # 创建圆
        ax.scatter(data[:, 0], data[:, 1], color='blue', marker='.', s=5)  # 绘制数据点
        ax.add_artist(circle)  # 绘制圆
        ax.scatter(centroid[0], centroid[1], color='red', s=5)  # 绘制圆心
        ax.set_aspect('equal', adjustable='datalim')  # 设置图片的纵横比和调整图形的方式

    plt.tight_layout()
    plt.show()


def main():
    """
    generate gbs and visualize gbs, gbs represents "Granular Balls" or "Granular Ball Space"

    :return:
    """
    # import data
    data = np.loadtxt('sample.txt')
    # generate gbs
    gbs = generate_granular_balls(data)
    # visualize gbs
    fig, ax = plt.subplots()
    plot_granular_balls(gbs, ax)


if __name__ == '__main__':
    main()
