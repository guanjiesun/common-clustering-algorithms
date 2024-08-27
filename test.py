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
        if len(gb) > 20:
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
            gb_list_not.append(gb)
        else:
            if get_radius(gb) <= 2 * radius_detect:
                gb_list_not.append(gb)
            else:
                ball_1, ball_2 = spilt_ball(gb)
                gb_list_temp.extend([ball_1, ball_2])

    return gb_list_temp, gb_list_not


def gbc(data):
    # 存储当前正在处理或待处理的粒球
    gb_list = [data]

    # 存储不能或不需要进一步分割的粒球
    gb_list_not = list()

    while True:
        # old_n表示粒球的数量
        old_n = len(gb_list) + len(gb_list_not)

        gb_list, gb_list_not = division(gb_list, gb_list_not)

        # new_n表示粒球的数量
        new_n = len(gb_list) + len(gb_list_not)

        if new_n == old_n:
            # 若new_n和old_n相等，则粒球生成完成，停止迭代
            gb_list = gb_list_not
            break

    # 获取radius_detect
    radii = list()
    for gb in gb_list:
        if len(gb) >= 2:
            radii.append(get_radius(gb))
    radius_median = np.median(radii)
    radius_mean = np.mean(radii)
    radius_detect = max(radius_median, radius_mean)

    # 对粒球进行规范会处理
    gb_list_not = list()
    while 1:
        old_n = len(gb_list) + len(gb_list_not)
        gb_list, gb_list_not = normalized_ball(gb_list, gb_list_not, radius_detect)
        new_n = len(gb_list) + len(gb_list_not)
        if new_n == old_n:
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
        centroid = np.mean(gb, axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)
    print(centroids.shape)
    plt.title('Granular Ball Centers')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=10, marker='.', color='red')
    plt.show()


def main():
    # 加载数据
    # folder_path = Path('./datasets')
    # csv_files = list(folder_path.glob("*.csv"))
    # for csv_file in csv_files:
    #     print(csv_file.name)
    #     dataset = pd.read_csv(csv_file, header=None).to_numpy()
    dataset = np.loadtxt('sample.txt')
    print(dataset.shape)
    gb_list = gbc(dataset)

    # 可视化原始数据和粒球
    visualize_original_data(dataset)
    visualize_gb_list(gb_list)


if __name__ == '__main__':
    main()
