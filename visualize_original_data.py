import numpy as np
import matplotlib.pyplot as plt


def visualize_original_data(data: np.ndarray) -> None:
    """可视化原始数据"""
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s=5, marker='.', color='black')
    ax.set_title('Original Data')
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def main():
    dataset = np.loadtxt('sample.txt')
    visualize_original_data(dataset)


if __name__ == '__main__':
    main()
