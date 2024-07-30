import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class DensityPeakClustering:
    """Density Peak Clustering Algorithm"""
    def __init__(self, data, dc, min_rho, min_delta):
        # TODO 初始化DPC算法的三个超参数
        self.data = data
        self.dc = dc
        self.min_rho = min_rho
        self.min_delta = min_delta

    def calculate_distances(self):
        pass

    def calculate_local_density(self):
        pass

    def calculate_delta(self):
        pass

    def find_cluster_centers(self):
        pass

    def assign_clusters(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
