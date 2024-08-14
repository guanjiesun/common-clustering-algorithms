from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_file_paths() -> list[Path]:
    """get corresponding path of each file"""
    # get the absolute path of current file using Path
    current_file_path = Path(__file__).resolve()
    folder_path = current_file_path.parent
    return list(folder_path.glob("*.csv"))


def visualize_datasets() -> None:
    """visualize 12 csv datasets from the datasets_from_gbsc folder"""
    # axs is np.ndarray, ndim=2, shape=(3, 4)
    fig, axs = plt.subplots(3, 4, figsize=(10, 8))
    axs: np.ndarray[plt.Axes] = axs.flatten()

    # get the absolute path of each file in the folder
    file_paths = get_file_paths()

    # visualize each csv file from the perspective of datapoint
    for i, file_path in enumerate(file_paths):
        # file_path: the absolute path of the corresponding csv file
        file_name = file_path.name
        data = pd.read_csv(file_path).to_numpy()
        axs[i].set_title(file_name)
        axs[i].scatter(data[:, 0], data[:, 1], s=5, color='blue', marker='.')
        axs[i].set_aspect('equal', 'datalim')

    plt.tight_layout()
    plt.show()


def main() -> None:
    """visualize 12 csv datasets"""
    visualize_datasets()


if __name__ == '__main__':
    main()
