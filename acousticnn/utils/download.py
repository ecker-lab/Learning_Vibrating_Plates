# PARTIALLY TAKEN FROM https://github.com/pdebench/PDEBench (MIT LICENSE)

import argparse
from torchvision.datasets.utils import download_url
import subprocess
import os


BASE_PATH = "https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/UWF7RB/"


class G5000:
    files = [
        ["APW6VO", "G5000_BCFP_1000_test.h5"],
        ["INRTIJ", "G5000_BCFP_2000_train.h5"],
        ["5XEC7Z", "G5000_BCFP_3000_train.h5"],
    ]

class V5000:
    files = [
        ["RFXBSR", "V5000_1000_test.h5"],
        ["F0IJ7E", "V5000_2000_train.h5"],
        ["2I5P0G", "V5000_3000_train.h5"]
    ]

class single_example_G5000:
    files = [["APW6VO", "G5000_BCFP_1000_test.h5"]]

class V50k:
    files = [
        ["V5YOL1", "V50000_15f.h5"]
    ]


def get_ids(name):
    datasets = {
        "G5000": G5000,
        "V5000": V5000,
        "single_example_G5000": single_example_G5000,
        'V50k': V50k
    }

    dataset = datasets.get(name)
    if dataset is not None:
        return dataset.files
    else:
        raise NotImplementedError (f"Dataset {name} does not exist.")


def download_data(root_folder, dataset_name):
    """
    Download data splits specific to a given setting.

    Args:
    root_folder: The root folder where the data will be downloaded
    dataset_name: The name of the dataset to download, one of: G5000, V5000 or single_example_G5000
    """

    print(f"Downloading data for {dataset_name} ...")

    # Load and parse metadata csv file
    files = get_ids(dataset_name)
    print(files)
    # Iterate ids and download the files
    for key, name in files:
        url = BASE_PATH + key
        print(url)
        #print(["wget", url, "-O", os.path.join(root_folder, name)])
        subprocess.run(["wget", "--no-check-certificate", url, "-O", os.path.join(root_folder, name)])

        #download_url(url, root_folder, name)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="Download Script",
        description="Helper script to download the Vibrating Plates datasets",
        epilog="",
    )

    arg_parser.add_argument(
        "--root_folder",
        type=str,
        required=True,
        help="Root folder where the data will be downloaded",
    )
    arg_parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset setting to download",
    )

    args = arg_parser.parse_args()

    download_data(args.root_folder, args.dataset_name)