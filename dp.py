import numpy as np
import matplotlib.pyplot as plt


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
    fig, ax = plt.subplots()
    ax.scatter(rho, delta, s=15, marker='.', color='black')
    ax.set_xlabel('rho')
    ax.set_ylabel('delta')
    ax.set_title('rho-delta')
    plt.savefig('dp_decision_graph.pdf', format='pdf')
    plt.show()


def main():
    distances = load_distance_matrix()
    dc = calculate_dc(distances)
    rho = calculate_local_density(distances, dc)
    delta, nearest_neighbor = calculate_delta(distances, rho)
    visualize_decision_graph(rho, delta)


if __name__ == "__main__":
    main()
