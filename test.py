import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


def spilt_ball(data):
    centroid = np.mean(data, axis=0)
    ball1, ball2 = list(), list()
    foo, foo_idx = 0, 0
    bar, bar_idx = 0, 0

    # 找到距离粒球质心最远的样本点foo_idx
    for i in range(len(data)):
        if foo < (sum((data[i] - centroid) ** 2)):
            foo = sum((data[i] - centroid) ** 2)
            foo_idx = i

    # 找到距离样本点foo_idx最远的样本点bar_idx
    for i in range(len(data)):
        if bar < (sum((data[i] - data[foo_idx]) ** 2)):
            bar = sum((data[i] - data[foo_idx]) ** 2)
            bar_idx = i

    # 以foo_idx和bar_idx为中心，将粒球所有的样本点划分为两个子集
    for j in range(0, len(data)):
        if (sum((data[j] - data[foo_idx]) ** 2)) < (sum((data[j] - data[bar_idx]) ** 2)):
            ball1.extend([data[j]])
        else:
            ball2.extend([data[j]])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]


def get_dm(gb: np.ndarray):
    """dm means distribution measure(和粒球的平均半径等价)"""
    num = len(gb)
    if num == 0:
        return 0

    # centroid是粒球的质心
    centroid = gb.mean(0)
    # mean_radius是粒球的平均半径(所有点到质心的平均距离)
    mean_radius = np.mean(pairwise_distances(gb, centroid.reshape(1, -1)))
    # 返回粒球gb的平均半径
    if num > 2:
        return mean_radius
    else:
        return 1


def division(gb_list, gb_list_not):
    """根据DM划分粒球"""
    gb_list_new = list()
    for gb in gb_list:
        if len(gb) > 10:
            ball_1, ball_2 = spilt_ball(gb)
            dm_parent, dm_child_1, dm_child_2 = get_dm(gb), get_dm(ball_1), get_dm(ball_2)
            w = len(ball_1) + len(ball_2)
            if w == 0:
                raise ZeroDivisionError("Waring!! w == 0 is True")
            if w != len(gb):
                raise TypeError("Warning!! w != len(gb) is True")
            w1, w2 = len(ball_1) / w, len(ball_2) / w
            w_child = w1 * dm_child_1 + w2 * dm_child_2

            # 判断粒球gb是否被分割
            if w_child < dm_parent:
                # 如果子粒球的加权平均半径小于父粒球，则接受分割
                gb_list_new.extend([ball_1, ball_2])
            else:
                # 否则，将粒球添加到不可分割列表
                gb_list_not.append(gb)
        else:
            gb_list_not.append(gb)
    return gb_list_new, gb_list_not


def get_radius(gb: np.ndarray):
    """计算粒球gb的半径"""
    # center是粒球的质心
    center = gb.mean(axis=0)
    # radius是粒球的半径((所有点到质心的最大距离))
    radius = np.max(pairwise_distances(gb, center.reshape(1, -1)))
    return radius


def normalized_ball(gb_list, gb_list_not, radius_detect):
    gb_list_temp = list()
    for gb in gb_list:
        if len(gb) < 2:
            # 如果粒球中的点少于2个，则将此粒球添加到不可分割的粒球列表中
            gb_list_not.append(gb)
        else:
            if get_radius(gb) <= 2 * radius_detect:
                # 如果粒球的半径小于等于检测半径的两倍，则将此粒球添加到不可分割的粒球列表中
                gb_list_not.append(gb)
            else:
                # 否则，将粒球分割成两个子粒球
                ball_1, ball_2 = spilt_ball(gb)
                # 将这两个子粒球添加到gb_list_temp列表中
                gb_list_temp.extend([ball_1, ball_2])

    return gb_list_temp, gb_list_not


def gbc(data):
    """首先根据数据的密度特性进行粗略划分，然后再基于一个全局的尺度（检测半径）进行细化，有助于产生更均匀和合理的聚类结果"""

    # 存储当前正在处理或待处理的粒球
    gb_list = [data]

    # 存储不能或不需要进一步分割的粒球
    gb_list_not = list()

    while True:
        """反复调用division函数，直到无法继续划分"""

        # 记录当前粒球的总数
        old_n = len(gb_list) + len(gb_list_not)

        # 进行粒球划分
        gb_list, gb_list_not = division(gb_list, gb_list_not)

        # 计算划分后的粒球总数
        new_n = len(gb_list) + len(gb_list_not)

        # 如果粒球数量没有变化，说明无法继续划分，退出循环
        if new_n == old_n:
            # 若new_n和old_n相等，则粒球生成完成，停止迭代
            gb_list = gb_list_not
            break

    # 计算检测半径
    radii = list()
    for gb in gb_list:
        if len(gb) > 1:
            radii.append(get_radius(gb))
    # 求radii的中位数
    radius_median = np.median(radii)
    # 求radii的平均值
    radius_mean = np.mean(radii)
    # 检测半径为radii的平均值和中位数中的较大值
    radius_detect = np.max([radius_median, radius_mean])

    # 重置不可分割粒球列表
    gb_list_not = list()
    # 基于检测半径的规范化
    while True:
        """计算一个检测半径，然后反复调用normalized_ball函数进行规范化，直到无法继续规范"""
        # 记录当前粒球的总数old_n
        old_n = len(gb_list) + len(gb_list_not)

        # 进行规范化
        gb_list, gb_list_not = normalized_ball(gb_list, gb_list_not, radius_detect)

        # 计算规范化后的粒球总数new_n
        new_n = len(gb_list) + len(gb_list_not)

        if new_n == old_n:
            # 如果粒球数量没有变化，说明无法继续规范化，退出循环
            gb_list = gb_list_not
            break

    return gb_list


def visualize_original_data(dataset: np.ndarray) -> None:
    """可视化原始数据"""

    plt.scatter(dataset[:, 0], dataset[:, 1], s=10, marker='.', color='black')
    plt.title('Original Data')
    plt.show()


def visualize_gb_list(gb_list):
    centroids = list()
    for gb in gb_list:
        # gb是一个二维numpy数组
        centroid = np.mean(gb, axis=0)
        centroids.append(centroid)

    # centroids的每一个元素是对应粒球的质心
    centroids = np.array(centroids)
    plt.title('Granular Ball Centroids')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=10, marker='.', color='red')
    plt.show()


class NaturalNeighborsSearch:

    def __init__(self, distances):
        self.A = distances  # 距离矩阵
        (self.NaN, self.nn, self.rnn, self.dis_index) = self.natural_search()

    def get_dis_index(self):
        # 获取带索引的排序距离字典
        distances = self.A
        n = distances.shape[0]
        dis_index = {}
        nn = {}
        rnn = {}
        for i in range(n):
            # 对每个点的距离进行排序
            dis = np.sort(distances[i, :])
            # 获取排序后的索引
            index = np.argsort(distances[i, :])
            dis_index[i] = [dis, index]
            # 初始化最近邻列表
            nn[i] = []
            # 初始化逆近邻列表
            rnn[i] = []
        return dis_index, nn, rnn

    def natural_search(self):
        # 自动迭代寻找自然邻居
        n = self.A.shape[0]
        dis_index, nn, rnn = self.get_dis_index()
        nb = [0]*n
        t = 0
        num_2 = 0
        while t+1 < n:
            for i in range(0, n):
                x = i
                y = dis_index[x][1][t+1]  # 获取第t+1近的邻居
                nn[x].append(y)  # 添加到最近邻列表
                rnn[y].append(x)  # 添加到逆近邻列表
                nb[y] += 1
            num_1 = nb.count(0)
            if num_1 != num_2:
                num_2 = num_1
            else:
                break
            t += 1

        natural_neighbors = []
        for i in range(len(nn)):
            # 计算自然邻居
            natural_neighbors.append(set(nn[i]) & set(rnn[i]))

        return natural_neighbors, nn, rnn, dis_index


class SampleDistribution:
    def __init__(self, data, label, data_index):
        self.data = data
        self.dataIndex = data_index
        self.center = self.data.mean(0)  # 计算中心
        self.label = label
        self.num = len(data)
        self.baryCenter = self.get_bary_center()

    def get_bary_center(self):
        # 获得样本分布的质心
        return self.dataIndex[np.argsort(((self.data - self.center) ** 2).sum(axis=1) ** 0.5)[0]]


class SDGS:
    def __init__(self, data, distance_matrix):
        self.data_ = data
        self.sdn_ = 0  # 样本分布数量
        self.distanceMatrix_ = distance_matrix
        self.nn_tool = NaturalNeighborsSearch(distance_matrix)
        self.MGN_ = self.mgb_generation()
        self.SDGS_ = self.sdgs_generation()

    def sdgs_label(self):
        # 为每个数据点分配标签
        labels = np.zeros(len(self.data_), dtype=int)
        for label, fdg in self.SDGS_.items():
            for index in fdg.dataIndex:
                labels[index] = label
        return list(labels)

    def mgb_generation(self):
        # 生成互助组（Mutual Group Neighbors）
        intersection_list = []
        for i, nan in enumerate(self.nn_tool.NaN):
            temp = set()
            temp |= nan | {i}
            ith_set = temp
            intersection_list.append(ith_set)
        mgn = intersection_list
        return mgn

    def sdgs_generation(self):
        # 生成样本分布组集（Sample Distribution Group Set）
        mgn_temp = copy.deepcopy(self.MGN_)

        def merge(mgn):
            # 合并互助组
            sdgs_ = {}
            iterate_list_ = [False] * len(mgn)
            distribution_count = 0
            for i in range(len(iterate_list_)):
                if not iterate_list_[i]:
                    sdgs_.setdefault(distribution_count, mgn[i])
                    for j in range(i, len(iterate_list_)):
                        if len(sdgs_[distribution_count] & mgn[j]) > 1 and not iterate_list_[j]:
                            iterate_list_[j] = True
                            sdgs_[distribution_count] |= mgn[j]
                    distribution_count += 1
            return sdgs_

        while True:
            sdgs = merge(mgn_temp)
            if len(sdgs) == len(mgn_temp):
                break
            mgn_temp = sdgs

        # 形成数据分布组
        ret_sdgs = {}
        count = 0
        for value in sdgs.values():
            real_data = []
            for v in value:
                real_data.append(self.data_[v])
            real_data = np.array(real_data)
            ret_sdgs[count] = SampleDistribution(real_data, count, list(value))
            count += 1

        self.sdn_ = len(ret_sdgs)
        return ret_sdgs


def visualize_clustering_result(centroids, labels):
    plt.scatter(centroids[:, 0], centroids[:, 1], s=10, marker='.', c=labels, cmap='plasma')
    plt.show()


class NARD:

    def __init__(self, data, nn_tool, sdgs):
        self.data_ = data
        self.centers_ = list()
        self.SDGS_ = sdgs.SDGS_
        self.SDGS_class_ = sdgs
        self.NNtool_ = nn_tool
        self.NNES_ = self.nnes_initiation()
        self.nard = self.nard_generation()

    # 实例化对象时自动进行邻域扩张
    def nnes_initiation(self):
        nnes = {}
        for i in range(len(self.NNtool_.NaN)):
            neighbor_range_temp = set(self.NNtool_.NaN[i])
            for extraNeighbor in self.NNtool_.NaN[i]:
                neighbor_range_temp |= set(self.NNtool_.NaN[extraNeighbor])
            neighbor_range_temp2 = set(neighbor_range_temp)
            for extraNeighbor in neighbor_range_temp:
                neighbor_range_temp2 |= set(self.NNtool_.NaN[extraNeighbor])
            if i in neighbor_range_temp2:
                neighbor_range_temp2.remove(i)
            nnes[i] = list(neighbor_range_temp2)
        return nnes

    # 采用局部相对密度的计算方式，来得到局部的自适应符合分布的密度，密度=自身值/局部最大值
    def nard_generation(self):
        ret_nard = np.zeros(len(self.data_))
        for key, value in self.SDGS_.items():
            nnes_mapped_of_fdg = []

            for i, index in enumerate(value.dataIndex):
                # 对下标进行映射，特征分布里面的NAN下标是全局的，计算时需要传局部的
                nnes_mapped = np.where(np.isin(value.dataIndex, self.NNES_[index]))[0]
                # 转换全局的index为局部的index，用于局部相对密度计算
                nnes_mapped_of_fdg.append(nnes_mapped)
            ndd = self.ndd_generation(value.data, nnes_mapped_of_fdg, value.dataIndex)
            local_density_peak = np.max(ndd)
            for i, index in enumerate(value.dataIndex):
                nard = ndd[i] / local_density_peak
                if nard == 1:
                    self.centers_.append(index)
                ret_nard[index] = nard
        return ret_nard

    def ndd_generation(self, fdg_data, adaptive_neighborhood, value_data_index):
        # 计算特征分布内数据的距离矩阵
        dis_matrix_of_fdg = pairwise_distances(fdg_data)

        length = len(fdg_data)
        dist, mgd = self.mgd_generation(dis_matrix_of_fdg, length, adaptive_neighborhood, value_data_index)
        ndd = list()
        for i in range(len(adaptive_neighborhood)):
            ndd_i = mgd[i]
            if len(adaptive_neighborhood[i]) > 0:
                for index, j in enumerate(adaptive_neighborhood[i]):  # for each neighbor
                    if j != -1:
                        wk_den_j = mgd[j] * (1 / dist[i][index])  # wk_den_j is an array
                        ndd_i = ndd_i + wk_den_j
                ndd.append(ndd_i)
            else:
                ndd.append(ndd_i)
        return ndd

    def mgd_generation(self, dis_matrix_of_fdg, length, adaptive_neighborhood, value_data_index):

        dist = list()
        mgd = list()
        for i in range(length):
            ith_distances = dis_matrix_of_fdg[i]
            dist.append(ith_distances[adaptive_neighborhood[i]])
            # 如果邻居数为0，则自适应邻居距离设置为无限小
            if len(adaptive_neighborhood[i]) != 0:
                mgd.append(1 / np.average(dist[i]))
            else:
                # 防止出现inf
                mgd.append(0.0000001)
        return dist, mgd


def main():
    # folder_path = Path('./datasets')
    # csv_files = list(folder_path.glob("*.csv"))
    # for csv_file in csv_files:
    #     print(csv_file.name)
    #     dataset = pd.read_csv(csv_file, header=None).to_numpy()
    # 加载数据
    dataset = np.loadtxt('sample.txt')
    print(dataset.shape)

    # gb_list的每一个原始都是一个二维numpy数组，每个数组表示一个粒球
    gb_list = gbc(dataset)

    # 可视化原始数据
    visualize_original_data(dataset)
    # 可视化基于原始数据生成的粒球空间
    visualize_gb_list(gb_list)

    # 获取由粒球质心组成的二维numpy数组
    centroids = list()
    for gb in gb_list:
        # gb是一个二维numpy数组
        centroid = np.mean(gb, axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    print(centroids.shape)

    # 获取粒球距离矩阵(两个粒球之间的距离用两个粒球之间质心的距离表示)
    distances = pairwise_distances(centroids)
    print(distances.shape)

    # 计算sdgs
    sdgs = SDGS(centroids, distances)
    sdgs_labels = sdgs.sdgs_label()
    sdgs_labels = np.array(sdgs_labels)
    print(np.unique(sdgs_labels))

    # 可视化sdgs产生的标签
    visualize_clustering_result(centroids, sdgs_labels)

    # 创建一个NaturalNeighborsSearch实例nn_tool
    nn_tool = NaturalNeighborsSearch(distances)

    # 创建一个NARD实例nard
    nard = NARD(centroids, nn_tool, sdgs)
    print(nard.nard.shape)
    print(nard.nard.max(), nard.nard.min())


if __name__ == '__main__':
    main()
