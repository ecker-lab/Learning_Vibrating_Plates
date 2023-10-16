# PARTIALLY TAKEN FROM https://github.com/pdebench/PDEBench (MIT LICENSE)

import argparse
from torchvision.datasets.utils import download_url
	
BASE_PATH = "https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/UWF7RB/"

class full_resolution_G5000:
    files = [
        ["8PWA5U", "full_resolution_g5000_1500_id_0.h5"],
        ["SM8ATR", "full_resolution_g5000_1500_id_1.h5"],
        ["VE9UXU", "full_resolution_test_g5000_500_id_0.h5"],
        ["NGIFRH", "full_resolution_test_g5000_500_id_1.h5"],
        ["AW1TSR"  "full_resolution_train_g5000_2000.h5"]
    ]

class full_resolution_V5000:
    files = [
        ["N5K05X", "full_resolution_test_V5000_1000.h5"],
        ["B30ZDJ",  "full_resolution_train_V5000_2500_id_0.h5"],
        ["YHBWGV",  "full_resolution_train_V5000_2500_id_1.h5"]
    ]

class reduced_resolution_G5000:
    files = [
        ["NYUBJR", "test_g5000_1000.h5"],
        ["6S2OZA", "train_g5000_1500_id_0.h5"],
        ["V9OCLZ", "train_g5000_1500_id_1.h5"],
        ["UWSRW0", "train_g5000_2000.h5"]
    ]

class reduced_resolution_V5000:
    files = [
        ["DG3EZP", "test_V5000_1000.h5"],
        ["RJWVTL", "train_V5000_2500_id_0.h5"],
        ["9NVYXS", "train_V5000_2500_id_1.h5"]
    ]

class single_example_G5000:
    files = [["NYUBJR", "test_g5000_1000.h5"]]


def get_ids(name):
    datasets = {
        "full_resolution_G5000": full_resolution_G5000,
        "full_resolution_V5000": full_resolution_V5000,
        "reduced_resolution_G5000": reduced_resolution_G5000,
        "reduced_resolution_V5000": reduced_resolution_V5000,
        "single_example_G5000": single_example_G5000
    }
    
    dataset = datasets.get(name)
    if dataset is not None:
        return dataset.files
    else:
        raise NotImplementedError (f"Dataset {name} does not exist.")


def download_data(root_folder, dataset_name):
    """ "
    Download data splits specific to a given setting.

    Args:
    root_folder: The root folder where the data will be downloaded
    dataset_name: The name of the dataset to download, one of: reduced_resolution_V5000, reduced_resolution_G5000,
       full_resolution_V5000, full_resolution_G5000 or single_example_G5000  """

    print(f"Downloading data for {dataset_name} ...")

    # Load and parse metadata csv file
    files = get_ids(dataset_name)

    # Iterate ids and download the files
    for id, name in files:
        url = BASE_PATH + id
        download_url(url, root_folder, name)


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