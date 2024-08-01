from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_csv_files_path(folder_name: str = 'datasets_from_gbsc') -> list[Path]:
    """
    get the path for each csv dataset

    :param folder_name: the folder contains 12 csv datasets
    :return:
    """
    # get the absolute path of current file using Path
    current_file_path = Path(__file__).resolve()
    folder_path = current_file_path.parent / folder_name
    return list(folder_path.glob("*.csv"))


def visualize_datasets() -> None:
    """
    visualize 12 csv datasets from the folder "datasets_from_gbsc"

    :return: None
    """
    # axs is np.ndarray, ndim=2, shape=(3, 4)
    fig, axs = plt.subplots(3, 4, figsize=(10, 8))
    axs: np.ndarray[plt.Axes] = axs.flatten()

    # get the absolute path of each file in the folder using folder_path.glob
    csv_file_paths = get_csv_files_path()

    # visualize each csv file from the perspective of datapoint
    for i, csv_file_path in enumerate(csv_file_paths):
        # csv_file_path: the absolute path of the corresponding csv file
        csv_file_name = csv_file_path.name
        data = pd.read_csv(csv_file_path).to_numpy()
        axs[i].set_title(csv_file_name)
        axs[i].scatter(data[:, 0], data[:, 1], s=5, color='blue')
        axs[i].set_aspect('equal', 'datalim')

    plt.tight_layout()
    plt.show()


def main():
    """
    visualize 12 datasets

    :return: None
    """
    visualize_datasets()


if __name__ == '__main__':
    main()
