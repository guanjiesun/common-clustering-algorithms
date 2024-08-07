import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import RectangleSelector, Cursor
from sklearn.metrics import pairwise_distances
from scipy.stats import gaussian_kde


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


def calculate_dc(distances: np.ndarray):
    upper_tri = distances[np.triu_indices(distances.shape[0], k=1)]
    sorted_distances = np.sort(upper_tri)
    n = len(sorted_distances)
    percent = 2.0
    position = int(np.round(n*percent / 100))
    dc = sorted_distances[position]
    return dc


def gaussian_kernel(d_i_j, dc):
    return np.exp(-np.square(d_i_j/dc))


def calculate_local_density(distances, dc):
    """calculate local density for each sample"""
    n_samples = len(distances)
    rho = np.zeros(n_samples)
    for i in range(n_samples-1):
        for j in range(i+1, n_samples):
            foo = gaussian_kernel(distances[i, j], dc)
            rho[i] += foo
            rho[j] += foo
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

        # 对于其他的点
        # 找到密度比其大的点的序号
        index_higher_rho = index_rho[:i]
        # 获取这些点距离当前点的距离,并找最小值
        delta[index] = np.min(distances[index, index_higher_rho])

        # 保存最近邻点的编号
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
    plt.colorbar(sc, label='Density')

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

    def on_key(event):
        if event.key == 'c':
            for patch in ax.patches:
                patch.remove()
            density_peak_indices.clear()
            fig.canvas.draw()

    rs = RectangleSelector(ax, select_callback, useblit=True,
                           button=[1, 3],  # disable middle button
                           minspanx=0, minspany=0,
                           spancoords='data',
                           interactive=True)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    return density_peak_indices


def assign_points_to_clusters(rho, centroids: list[int], nearest_neighbor):
    n_samples = np.shape(rho)[0]
    labels = np.full(n_samples, -1)

    for i, centroid in enumerate(centroids):
        labels[centroid] = i

    index_rho = np.argsort(rho)[::-1]
    for i, index in enumerate(index_rho):
        # 从密度大的点进行标号
        if labels[index] == -1:
            # 如果没有被标记过，那么聚类标号与距离其最近且密度比其大的点的标号相同
            labels[index] = labels[int(nearest_neighbor[index])]

    return labels


def draw_cluster(data, labels, centroids, dic_colors):
    plt.cla()
    k = np.shape(centroids)[0]

    for i in range(k):
        sub_index = np.where(labels == i)[0]
        sub_delta = data[sub_index]
        # 画数据点
        plt.scatter(sub_delta[:, 0], sub_delta[:, 1], s=15, color=dic_colors[i])
        # 画聚类中心
        plt.scatter(data[centroids[i], 0], data[centroids[i], 1], color='black', marker="x", s=100)
    plt.show()


def main():
    # data = pd.read_csv('./datasets_from_gbsc/D1.csv').to_numpy()
    # distances = load_distance_matrix()
    data = np.loadtxt('example_data.txt')
    distances = pairwise_distances(data)
    dc = calculate_dc(distances)
    rho = calculate_local_density(distances, dc)
    delta, nearest_neighbor = calculate_delta(distances, rho)
    density_peak_points = visualize_decision_graph(rho, delta)
    labels = assign_points_to_clusters(rho, density_peak_points, nearest_neighbor)
    dic_colors = {0: (.8, 0, 0), 1: (0, .8, 0),
                  2: (0, 0, .8), 3: (.8, .8, 0),
                  4: (.8, 0, .8), 5: (0, .8, .8),
                  6: (0, 0, 0)}
    draw_cluster(data, labels, density_peak_points, dic_colors)


if __name__ == "__main__":
    main()
