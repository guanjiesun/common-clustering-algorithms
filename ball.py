import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class GranularBall:
    """粒球"""
    def __init__(self, data: np.ndarray):
        """粒球初始化，data是一个ndarray，形状是MxN，维度数是2"""
        self.data = data
        self.centroid = np.mean(self.data, axis=0)  # 粒球的质心
        # self.radius = np.mean(np.sqrt(np.sum(np.square(self.data - self.centroid), axis=1)))  # 欧式距离的平均值
        self.radius = np.max(np.sqrt(np.sum(np.square(self.data - self.centroid), axis=1)))  # 欧式距离的最大值

    def info(self):
        """打印粒球信息"""
        print(f"centroid = {self.centroid}\nradius = {self.radius}")


def plot_ball(gb: GranularBall, ax: plt.Axes) -> None:
    """绘制数据点和粒球"""
    data, centroid, radius = gb.data, gb.centroid, gb.radius  # 获取粒球包含的数据、粒球形心和粒球半径
    circle = Circle(centroid, radius, fill=False, color='red')  # 创建圆
    ax.scatter(data[:, 0], data[:, 1], color='black', marker='.', s=10)  # 绘制数据点
    ax.add_artist(circle)  # 绘制圆
    ax.scatter(centroid[0], centroid[1], color='red', s=30)  # 绘制圆心
    ax.set_xlim(-5, 5)
    ax.set_yticks(np.arange(-5, 5))
    ax.set_ylim(-5, 5)
    ax.set_yticks(np.arange(-5, 5))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Data Points with Fitted Circle")
    # aspect ratio表示纵横比或者长宽比
    # 纵横比的参数设置为equal时，它确保x轴和y轴的刻度是相等的。这意味着在图上，相同的数据单位在x和y方向上会占用相同的物理长度
    # adjustable参数的值有box和datalim
    ax.set_aspect('equal', adjustable='box')  # 设置图片的纵横比和调整图形的方式（以满足纵横比）


def configure_figure() -> None:
    # 设置图片的属性
    plt.tight_layout()
    plt.show()


def main():
    # 生成数据
    n_samples, n_features = 1000, 2
    np.random.seed(n_samples)
    data = np.random.randn(n_samples, n_features)
    # 实例化一个粒球对象
    gb = GranularBall(data)
    gb.info()
    # 创建图片
    fig, ax = plt.subplots(figsize=(8, 6))
    # 粒球可视化
    plot_ball(gb, ax)
    # 设置图片属性
    configure_figure()


def plot_datasets_from_gbsc():
    """画出datasets_from_gbsc文件夹中的12个数据集"""
    fig, axs = plt.subplots(3, 4, figsize=(10, 8))  # axs是一个二维数组，形状(3, 4)
    axs = axs.flatten()  # 将3x4的二维数组展平为(12,)的一维数组，便于索引

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
        axs[i].set_xlim(0, 6)  # 设置 x 轴范围
        axs[i].set_xticks(np.arange(0, 6.1, 1))  # 设置 x 轴刻度


if __name__ == '__main__':
    plot_datasets_from_gbsc()
    # configure_figure()
