import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


def spilt_ball(data):
    centroid = np.mean(data, axis=0)
    ball1, ball2 = list(), list()
    foo, foo_idx = 0, 0
    bar, bar_idx = 0, 0
    for i in range(len(data)):
        if foo < (sum((data[i] - centroid) ** 2)):
            foo = sum((data[i] - centroid) ** 2)
            foo_idx = i
    for i in range(len(data)):
        if bar < (sum((data[i] - data[foo_idx]) ** 2)):
            bar = sum((data[i] - data[foo_idx]) ** 2)
            bar_idx = i
    for j in range(0, len(data)):
        if (sum((data[j] - data[foo_idx]) ** 2)) < (sum((data[j] - data[bar_idx]) ** 2)):
            ball1.extend([data[j]])
        else:
            ball2.extend([data[j]])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]


def get_dm(gb):
    """计算每一个粒球的平均半径"""
    num = len(gb)
    if num == 0:
        return 0
    # center是粒球的质心
    center = gb.mean(0)
    # mean_radius是粒球的平均半径
    mean_radius = np.mean(pairwise_distances(gb, center.reshape(1, -1)))
    if num > 2:
        return mean_radius
    else:
        return 1


def division(gb_list, gb_list_not):
    gb_list_new = []
    for gb in gb_list:
        if len(gb) > 20:
            ball_1, ball_2 = spilt_ball(gb)
            dm_parent, dm_child_1, dm_child_2 = get_dm(gb), get_dm(ball_1), get_dm(ball_2)
            w = len(ball_1) + len(ball_2)
            if w == 0:
                raise ZeroDivisionError("Waring!! w == 0 is True")
            w1, w2 = len(ball_1) / w, len(ball_2) / w
            w_child = w1 * dm_child_1 + w2 * dm_child_2
            if w_child < dm_parent:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_not.append(gb)
        else:
            gb_list_not.append(gb)
    return gb_list_new, gb_list_not


def get_radius(gb):
    # center是粒球的质心
    center = gb.mean(axis=0)
    # radius是粒球的半径
    radius = np.max(pairwise_distances(gb, center.reshape(1, -1)))

    return radius


def normalized_ball(gb_list, gb_list_not, radius_detect):
    gb_list_temp = []
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
    gb_list_temp = [data]
    gb_list_not_temp = []

    # divide by DM
    while 1:
        ball_number_old = len(gb_list_temp) + len(gb_list_not_temp)
        gb_list_temp, gb_list_not_temp = division(gb_list_temp, gb_list_not_temp)
        ball_number_new = len(gb_list_temp) + len(gb_list_not_temp)
        if ball_number_new == ball_number_old:
            gb_list_temp = gb_list_not_temp
            break

    radius = []
    for gb in gb_list_temp:
        if len(gb) >= 2:
            radius.append(get_radius(gb))
    radius_median = np.median(radius)
    radius_mean = np.mean(radius)
    radius_detect = max(radius_median, radius_mean)
    gb_list_not_temp = []
    while 1:
        ball_number_old = len(gb_list_temp) + len(gb_list_not_temp)
        gb_list_temp, gb_list_not_temp = normalized_ball(gb_list_temp, gb_list_not_temp, radius_detect)
        ball_number_new = len(gb_list_temp) + len(gb_list_not_temp)
        if ball_number_new == ball_number_old:
            gb_list_temp = gb_list_not_temp
            break

    return gb_list_temp


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
    dataset = pd.read_csv('./datasets/D7.csv', header=None).to_numpy()
    # dataset = np.loadtxt('sample.txt')
    print(dataset.shape)
    gb_list = gbc(dataset)

    # 可视化原始数据和粒球
    visualize_original_data(dataset)
    visualize_gb_list(gb_list)


if __name__ == '__main__':
    main()
