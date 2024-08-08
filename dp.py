import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.stats import gaussian_kde
from matplotlib.widgets import RectangleSelector, Cursor
from matplotlib.backend_bases import MouseButton


def load_distance_matrix(file_path='example_distances.dat'):
    # 读取文件并提取数据
    data = np.loadtxt(file_path)
    # 获取点的数量
    num_points = int(max(data[:, :2].max(), data[:, :2].min()))

    # 创建一个空的距离矩阵
    distance_matrix = np.zeros((num_points, num_points))

    # 填充距离矩阵
    for row in data:
        i, j, distance = int(row[0]) - 1, int(row[1]) - 1, row[2]
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

    # 将对角线设置为0
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix


def calculate_dc(distances: np.ndarray) -> float:
    upper_tri = distances[np.triu_indices(distances.shape[0], k=1)]
    sorted_distances = np.sort(upper_tri)
    n = len(sorted_distances)

    percent = 2.0
    position = int(np.round(n*percent / 100))
    dc = sorted_distances[position]

    return float(dc)


def gaussian_kernel(d_i_j, dc):
    return np.exp(-np.square(d_i_j/dc))


def calculate_local_density(distances: np.ndarray, dc: float):
    """calculate local density for each sample"""
    n_samples = len(distances)
    rho = np.zeros(n_samples)
    for i in range(n_samples-1):
        for j in range(i+1, n_samples):
            rho[i] += gaussian_kernel(distances[i, j], dc)
            rho[j] += gaussian_kernel(distances[i, j], dc)
    return rho


def calculate_delta(distances: np.ndarray, rho):
    n_samples = len(distances)

    delta = np.zeros(n_samples)
    nearest_neighbor = np.full(n_samples, 0)
    index_rho = np.argsort(rho)[::-1]  # rho降序排序后的数组的元素在rho中的索引

    for i, index in enumerate(index_rho):
        # 对于密度最大的点
        if i == 0:
            continue

        # 对于其他的点，找到密度比其大的点的序号
        index_higher_rho = index_rho[:i]
        # 获取这些点距离当前点的距离，并找最小值
        delta[index] = np.min(distances[index, index_higher_rho])

        # 保存最近邻点的索引
        index_nn = np.argmin(distances[index, index_higher_rho])
        nearest_neighbor[index] = index_higher_rho[index_nn].astype(int)

    delta[index_rho[0]] = np.max(delta)
    return delta, nearest_neighbor


def visualize_decision_graph(rho, delta):
    density_peak_indices = list()  # 改名以反映它现在存储索引
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

    xy = np.vstack([rho, delta])
    z = gaussian_kde(xy)(xy)
    sc = ax.scatter(rho, delta, c=z, s=20, cmap='viridis')
    plt.colorbar(sc, label='Local Density')

    ax.set_xlabel('rho')
    ax.set_ylabel('delta')
    ax.set_title('rho-delta Decision Graph')
    ax.grid(True, linestyle='--', alpha=0.7)

    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

    def select_callback(e_click, e_release):
        x1, y1 = e_click.xdata, e_click.ydata
        x2, y2 = e_release.xdata, e_release.ydata
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # 选择区域内的点的索引
        mask = (rho >= x_min) & (rho <= x_max) & (delta >= y_min) & (delta <= y_max)
        selected_indices = np.where(mask)[0]  # 获取满足条件的索引

        density_peak_indices.extend(selected_indices)

        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             fill=False, edgecolor='r', linestyle='--')
        ax.add_patch(rect)
        fig.canvas.draw()
        print(f"Selected {len(selected_indices)} points")

    rs = RectangleSelector(ax, select_callback, useblit=True, button=[MouseButton.LEFT, MouseButton.RIGHT],
                           minspanx=0, minspany=0, spancoords='data', interactive=True)

    plt.show()

    return density_peak_indices


def assign_points_to_clusters(rho, centroids: list[int], nearest_neighbor):
    n_samples = len(rho)

    # 初始化所有样本呢的簇标签为1
    labels = np.full(n_samples, -1)

    for i, centroid in enumerate(centroids):
        # 给聚类中心分配簇标签
        labels[centroid] = i

    index_rho = np.argsort(rho)[::-1]
    for i, index in enumerate(index_rho):
        # 从密度大的点进行标号
        if labels[index] == -1:
            # 如果没有被标记过，那么聚类标号与距离其最近且密度比其大的点的标号相同
            labels[index] = labels[int(nearest_neighbor[index])]

    return labels


def visualize_clustering_result(data, labels, centroids):
    # k表示找到的聚类中心的数量
    k = np.shape(centroids)[0]
    colors = sns.color_palette("husl", k)

    for i in range(k):
        sub_index = np.where(labels == i)[0]
        sub_delta = data[sub_index]
        # 画数据点
        plt.scatter(sub_delta[:, 0], sub_delta[:, 1], s=15, color=colors[i], marker='.')
        # 画聚类中心
        plt.scatter(data[centroids[i], 0], data[centroids[i], 1], color='black', marker="x", s=100)

    # 判断是否存在未被分配簇标签的样本(簇标签还是-1)，将这些样本画成黑色
    idx = np.where(labels == -1)[0]
    if list(idx):
        other_data = data[idx]
        plt.scatter(other_data[:, 0], other_data[:, 1], color='black', marker=".", s=30)
    plt.show()


def main():
    data = pd.read_csv('./datasets_from_gbsc/D2.csv').to_numpy()
    # distances = load_distance_matrix()
    # data = np.loadtxt('sample.txt')
    distances = pairwise_distances(data)

    # 计算截断距离
    dc = calculate_dc(distances)

    # 计算局部密度
    rho = calculate_local_density(distances, dc)

    # 计算相对距离和每个样本的最近邻
    delta, nearest_neighbor = calculate_delta(distances, rho)

    # 从决策图中选择密度峰值(聚类中心)
    centroids = visualize_decision_graph(rho, delta)

    # 分配非聚类中心并且返回每个样本的簇标签
    labels = assign_points_to_clusters(rho, centroids, nearest_neighbor)

    # 可视化聚类结果
    visualize_clustering_result(data, labels, centroids)


if __name__ == "__main__":
    main()
