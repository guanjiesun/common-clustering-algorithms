# Standard library/module imports
import os
from collections import deque

# Third-party library/module imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# local module imports
from kmeans import kmeans


class GranularBall:
    """粒球"""
    def __init__(self, data: np.ndarray):
        """
        粒球初始化

        :param data: np.ndarray, shape=(n_samples, m_features)
        """
        self.data = data
        # centroid.shape=(m_features,), centroid is an instance of np.ndarray
        self.centroid = np.mean(self.data, axis=0)
        # the mean value of the distances from all points to centroid
        # self.radius = np.mean(np.sqrt(np.sum(np.square(self.data - self.centroid), axis=1)))
        # the max value of the distances from all points to centroid
        self.radius = np.max(np.sqrt(np.sum(np.square(self.data - self.centroid), axis=1)))
        self.size = self.data.shape[0]


def generate_granular_balls(data: np.ndarray) -> list[GranularBall]:
    """
    基于一个数据集，生成一个粒球列表GBs

    :param data: np.ndarray, shape=(n_samples, m_features)
    :return: gbs, a list of Granular Balls, abbreviated as gbs
    """
    n = data.shape[0]  # 粒球中包含的样本数量
    threshold = np.sqrt(n)
    gbs = list()  # 初始化粒球列表
    queue = deque([GranularBall(data)])  # 初始化队列；deque函数需要的参数是一个可迭代对象
    while queue:
        gb = queue.popleft()  # 使用popleft而不是pop是因为保持queue先进先出的特性
        if gb.size > threshold:
            # 如果gb太大，则使用2means算法将gb划分为两个更小的粒球，并且将它们加入queue
            _, labels = kmeans(gb.data, 2)
            micro_gb_1 = GranularBall(gb.data[labels == 0])
            micro_gb_2 = GranularBall(gb.data[labels == 1])
            queue.extend([micro_gb_1, micro_gb_2])
        else:
            gbs.append(gb)

    return gbs


def plot_granular_balls(gbs: list[GranularBall], ax: plt.Axes) -> None:
    """绘制数据点和粒球"""
    for i, gb in enumerate(gbs):
        data, centroid, radius = gb.data, gb.centroid, gb.radius  # 获取粒球包含的数据、粒球形心和粒球半径
        circle = Circle(centroid, radius, fill=False, color='blue')  # 创建圆
        ax.scatter(data[:, 0], data[:, 1], color='blue', marker='.', s=10)  # 绘制数据点
        ax.add_artist(circle)  # 绘制圆
        ax.scatter(centroid[0], centroid[1], color='red', s=10)  # 绘制圆心
        ax.set_aspect('equal', adjustable='datalim')  # 设置图片的纵横比和调整图形的方式（以满足纵横比）
        # # 在圆心上添加数字标签
        # ax.text(centroid[0], centroid[1], str(i+1), color='black',
        #         fontweight='bold', ha='center', va='center')
    plt.show()


def plot_datasets_from_gbsc():
    """画出datasets_from_gbsc文件夹中的12个数据集"""
    fig, axs = plt.subplots(3, 4, figsize=(10, 8))  # axs是一个二维np.ndarray实例，形状(3, 4)
    # 将3x4的二维数组展平为(12,)的一维np.ndarray实例，便于索引
    # 需要使用类型注释，说明axs是一个np.ndarray实例，每一个元素都是plt.Axes实例
    axs: np.ndarray[plt.Axes] = axs.flatten()

    # os.path.abspath(__file__)获取当前文件的绝对路径
    # os.path.dirname返回给定路径的目录名部分（函数会去除文件名部分）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, 'datasets_from_gbsc')
    # os.listdir列出给定目录下的所有子目录和文件，返回一个列表
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]  # 只处理.csv文件
    # for循环处理csv文件
    for i, filename in enumerate(csv_files[:12]):  # 只处理前12个CSV文件
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path)
        axs[i].scatter(data.iloc[:, 0], data.iloc[:, 1], s=10)  # 绘制散点图
        axs[i].set_title(f'Dataset {i + 1}')  # 设置子图标题
        axs[i].set_aspect('equal', 'box')  # 保持纵横比
    plt.show()


def main():
    # 导入数据
    data = np.loadtxt('sample.txt')
    # 生成粒球列表gbs
    gbs = generate_granular_balls(data)
    # 画出生成的粒球
    fig, ax = plt.subplots()
    plot_granular_balls(gbs, ax)


if __name__ == '__main__':
    main()
