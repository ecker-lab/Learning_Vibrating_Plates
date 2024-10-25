import numpy as np
import os
import h5py
import hdf5plugin
import torch
import torch.nn.functional as F
compression = hdf5plugin.Blosc(cname='lz4', clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE)


def load_original_data(path):
    with h5py.File(path, 'r') as f:
        z_vel_abs = f["Results"]["z_vel_abs"][:]
        z_vel_mean_sq = f["Results"]["z_vel_mean_sq"][:]
        bead_patterns = f["Variation"]["Beading_pattern"][:]
        if "Phy_para" in f["Variation"]:
            phy_para = f["Variation"]["Phy_para"][:]
        else:
            phy_para = np.full((bead_patterns.shape[0], 4), [0.75, 0.5, 0.003, 0.02])

        # Load the additional datasets
        z_vel_sum_level = f["Results"]["z_vel_sum_level"][:]
        n_ellipse = f["Variation"]["n_ellipse"][:]
        n_lines = f["Variation"]["n_lines"][:]
        frequencies = f["Variation"]["f_samples"][:]

    return z_vel_abs, z_vel_mean_sq, bead_patterns, phy_para, z_vel_sum_level, n_ellipse, n_lines, frequencies


def reduce_resolution(z_vel_abs, resolution=(40, 60)):
    z_vel_abs = F.interpolate(torch.from_numpy(z_vel_abs), resolution, align_corners=True, mode="bilinear")
    z_vel_abs = z_vel_abs.numpy().astype(np.float32)
    return z_vel_abs


def save_data(path, z_vel_abs, z_vel_mean_sq, bead_patterns, phy_para, z_vel_sum_level, n_ellipse, n_lines, frequencies, overwrite=False):
    if not overwrite:
        assert not os.path.exists(path)
    with h5py.File(path, 'w') as f:
        f.create_dataset("bead_patterns", data=bead_patterns, **compression)
        f.create_dataset("z_vel_mean_sq", data=z_vel_mean_sq, **compression)
        f.create_dataset("z_vel_abs", data=z_vel_abs, **compression)
        if phy_para is not None:
            f.create_dataset("phy_para", data=phy_para, **compression)

        # Save the additional datasets
        f.create_dataset("z_vel_sum_level", data=z_vel_sum_level, **compression)
        f.create_dataset("n_ellipse", data=n_ellipse, **compression)
        f.create_dataset("n_lines", data=n_lines, **compression)
        f.create_dataset("frequencies", data=frequencies, **compression)


def add_moments_to_hdf5(path):
    import torch
    eps=1e-12
    with h5py.File(path, 'r+') as f:
        data = torch.from_numpy(f["z_vel_abs"][:])
        data = torch.log(torch.square(data) + eps)
        field_mean = torch.mean(data, axis=(0, 2, 3)).view(1, -1, 1, 1).numpy()
        field_std = torch.std(data - field_mean).numpy()

        f.create_dataset("field_mean", data=field_mean, **compression)
        f.create_dataset("field_std", data=field_std)
        print("added_moment")


def concatenate_h5_files(file_list, output_file):
    assert not os.path.exists(output_file)
    # Initialize empty lists to store concatenated data
    bead_patterns_list = []
    z_vel_mean_sq_list = []
    z_vel_abs_list = []
    phy_para_list = []

    # Read each file and append data to lists
    for file_path in file_list:
        with h5py.File(file_path, 'r') as f:
            bead_patterns_list.append(f["bead_patterns"][:])
            z_vel_mean_sq_list.append(f["z_vel_mean_sq"][:])
            z_vel_abs_list.append(f["z_vel_abs"][:])
            phy_para_list.append(f["phy_para"][:])

    # Concatenate along the first axis
    bead_patterns_concat = np.concatenate(bead_patterns_list, axis=0)
    z_vel_mean_sq_concat = np.concatenate(z_vel_mean_sq_list, axis=0)
    z_vel_abs_list = np.concatenate(z_vel_abs_list, axis=0)
    phy_para_concat = np.concatenate(phy_para_list, axis=0)

    # Save concatenated data to new h5 file
    with h5py.File(output_file, 'w') as f:
        f.create_dataset("bead_patterns", data=bead_patterns_concat, **compression)
        f.create_dataset("z_vel_mean_sq", data=z_vel_mean_sq_concat, **compression)
        f.create_dataset("z_vel_abs", data=z_vel_abs_list, **compression)
        f.create_dataset("phy_para", data=phy_para_concat, **compression)


def eval_file(path):
    with h5py.File(path, 'r') as f:
        print(f.keys())
        print(f["bead_patterns"])
        print(f["z_vel_mean_sq"])
        print(f["z_vel_abs"])
        print(f["phy_para"])


import h5py

def rename_dataset(filename, old_name, new_name):
    """
    Renames a dataset within an HDF5 file while preserving its properties and attributes.

    Args:
    filename (str): The path to the HDF5 file.
    old_name (str): The current name of the dataset.
    new_name (str): The new name for the dataset.
    """
    # Open the HDF5 file
    with h5py.File(filename, 'a') as file:
        # Ensure the old dataset exists
        if old_name not in file:
            raise ValueError(f"Dataset {old_name} not found in file.")

        # Ensure the new dataset name does not already exist
        if new_name in file:
            raise ValueError(f"Dataset {new_name} already exists in file.")

        # Access the old dataset
        old_ds = file[old_name]

        # Create a new dataset with the same settings as the old one
        new_ds = file.create_dataset(new_name, data=old_ds[...], **compression)

        # Delete the old dataset
        del file[old_name]


def add_frequencies_dataset(hdf5_path):
    with h5py.File(hdf5_path, 'a') as file:  # Open the file in append mode
        if 'z_vel_mean_sq' not in file:
            raise ValueError("Dataset 'z_vel_mean_sq' not found in the file.")

        num_rows = file['z_vel_mean_sq'].shape[0]
        frequencies = np.tile(np.arange(1, 301), (num_rows, 1))
        #print(frequencies.shape)
        # Ensure it does not already exist
        if 'frequencies' in file:
            raise ValueError("Dataset 'frequencies' already exists in the file.")

        file.create_dataset('frequencies', data=frequencies)
