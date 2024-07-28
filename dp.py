import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class DensityPeakClustering:
    # 密度峰值聚类算法DPC
    def __init__(self, data, dc, min_rho, min_delta):
        # TODO 初始化DPC算法的三个超参数
        self.data = data
        self.dc = dc
        self.min_rho = min_rho
        self.min_delta = min_delta

    def calculate_distances(self):
        # TODO 计算数据集中任意两个点之间的欧式距离
        n = self.data.shape[0]  # 数据点的数量
        distances = np.zeros((n, n))  # 初始化距离矩阵
        for i in range(n):
            for j in range(n):
                # 计算两点之间每个维度的差
                diff = self.data[i] - self.data[j]
                # 计算差的平方和
                squared_distance = np.sum(diff ** 2)
                # 取平方根得到欧氏距离
                distances[i, j] = np.sqrt(squared_distance)
        return distances

    def calculate_local_density(self):
        # TODO 计算数据集中每一个点的局部密度
        # 对于每个数据点，计算与其距离小于dc的邻近点数量，这个数量就是该点的局部密度
        distances = self.calculate_distances()
        number = distances.shape[0]
        # 初始化一个数组来存储每个点的局部密度
        local_densities = np.zeros(number)
        # 根据截断距离这个参数，计算每一个点的局部密度
        for i in range(number):
            # 获取当前点到所有其他点的距离
            distances_to_other_points = distances[i]
            # 计算在截断距离dc内的点的数量(邻近点的数量)。注意：排除了距离为0的情况（即点自身)
            # &是按位与运算符
            nearby_points = np.sum((distances_to_other_points < self.dc) & (distances_to_other_points > 0))
            # 将这个数量作为当前点的局部密度
            local_densities[i] = nearby_points
        return local_densities

    def calculate_delta(self):
        """TODO 计算每一个点的delta值和每一个点的最近邻
        输入：rho是一维数组，存储每一个点的局部密度
        输入：distances是二维数组，存储每一个点到其它点的欧式距离
        输出：delta是一维数组，存储每一个点的delta值
        输出：nearest_neighbor是一维数组，存储每一个点的最近邻
        """
        distances = self.calculate_distances()
        rho = self.calculate_local_density()
        n = len(rho)
        delta = np.zeros(n)
        nearest_neighbor = np.zeros(n, dtype=np.int64)

        for i in range(n):
            if rho[i] == max(rho):
                # 如果该点的局部密度就是数据集所有点中最大的，那么
                delta[i] = max(distances[i])  # 该点的delta值=该点到其它点的最大距离
                nearest_neighbor[i] = i  # 该点的最近邻就是自己
            else:
                """为什么使用 [0]：
                np.where函数返回一个元组，即使只有一个数组。
                [0]用于提取这个元组中的第一个（也是唯一的）数组。
                """
                # higher_density_indices存储局部密度比当前点的局部密度大的那些点的索引
                higher_density_indices = np.where(rho > rho[i])[0]
                # 从 distances的第i行中选择higher_density_indices指定的列
                # 该点的delta值=从比该点局部密度大的所有点中，离该点最近的点的距离就是该点的delta值
                delta[i] = min(distances[i, higher_density_indices])
                # 该点的最近邻=比该点局部密度大的所有点中，距离该点最近的点就是该点的最近邻
                nearest_neighbor[i] = higher_density_indices[np.argmin(distances[i, higher_density_indices])]
        return delta, nearest_neighbor

    def find_cluster_centers(self):
        """TODO 找到数据集的聚类中心
        delta值的设置是为了量化每个点相对于其他高密度区域的独立性。它与密度rho一起，
        为识别聚类中心、确定聚类数量和理解数据整体结构提供了关键信息
        1. rho值高：该点周围有许多其他点（高密度区域）
        2. delta值高：该点远离其他高密度点
        3. 聚类中心通常出现在决策图的右上角
        """
        # 返回聚类中心的索引（聚类中心也是从数据集中的点挑选出来的）
        # 选择那些rho和delta都很大的点作为聚类中心
        rho = self.calculate_local_density()
        delta, _ = self.calculate_delta()
        return np.where((rho > self.min_rho) & (delta > self.min_delta))[0]

    def assign_clusters(self):
        # TODO 将数据集中的点分配到相应的簇中
        n = len(self.data)
        cluster_centers = self.find_cluster_centers()
        _, nearest_neighbor = self.calculate_delta()
        # -1通常用来表示未分配任何簇的标签或噪声点
        labels = -np.ones(n, dtype=np.int64)
        for i, center in enumerate(cluster_centers):
            # 为每个聚类中心center分配一个唯一的簇标签i
            labels[center] = i

        for i in range(n):
            if labels[i] == -1:
                labels[i] = labels[nearest_neighbor[i]]
        print(f"未分配标签的点数: {np.sum(labels == -1)}")
        print(f"聚类中心的个数: {len(cluster_centers)}")
        return labels


def main():
    # 生成示例数据
    data, y = make_blobs(n_samples=500, centers=3, cluster_std=0.30, random_state=0)
    # data = np.loadtxt('sample.txt')
    # 初始化密度峰值聚类的三个超参数
    dc = 0.8  # 截断距离distance cutoff
    min_rho = 160  # 最小局部密度local density
    min_delta = 3.0  # 最小delta值
    # 实例化一个DPC算法
    dpc = DensityPeakClustering(data, dc, min_rho, min_delta)
    rho = dpc.calculate_local_density()
    delta, _ = dpc.calculate_delta()
    labels = dpc.assign_clusters()

    # 可视化结果
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制聚类结果
    scatter = ax0.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10')
    # 设置标题和标签
    ax0.set_title('Clustering Result', fontsize=14)
    ax0.set_xlabel('Feature 1', fontsize=12)
    ax0.set_ylabel('Feature 2', fontsize=12)
    # 添加网格线
    ax0.grid(True, linestyle='--', alpha=0.7)
    # 添加图例
    legend1 = ax0.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
    ax0.add_artist(legend1)

    # 绘制决策图
    ax1.scatter(rho, delta, alpha=0.8)
    ax1.set_xlabel('rho')
    ax1.set_ylabel('delta')
    ax1.set_title('Decision Graph')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
