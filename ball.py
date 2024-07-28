import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class GranularBall:
    """粒球"""
    def __init__(self, data: np.ndarray):
        """粒球初始化，data是一个ndarray，形状是MxN，维度数是2"""
        self.data = data
        self.centroid = np.mean(self.data, axis=0)  # 粒球的质心
        self.radius = np.mean(np.sqrt(np.sum(np.square(self.data - self.centroid), axis=1)))  # 欧式距离的平均值

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
    # aspect ratio表示纵横比或者长宽比
    # 纵横比的参数设置为equal时，它确保x轴和y轴的刻度是相等的。这意味着在图上，相同的数据单位在x和y方向上会占用相同的物理长度
    # adjustable参数的值有box和datalim
    ax.set_aspect('equal', adjustable='box')  # 设置图片的纵横比和调整图形的方式（以满足纵横比）


def configure_figure() -> None:
    # 设置图片的属性
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Data Points with Fitted Circle")
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
    fig, ax = plt.subplots(figsize=(6, 6))
    # 粒球可视化
    plot_ball(gb, ax)
    # 设置图片属性
    configure_figure()


if __name__ == '__main__':
    main()
