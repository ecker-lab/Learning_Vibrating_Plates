import numpy as np
import os
import h5py
import hdf5plugin
import torch
import torch.nn.functional as F
compression = hdf5plugin.Blosc(cname='lz4', clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE)


def load_original_data(path):
    with h5py.File(path, 'r') as f:
        z_abs_velocity = f["Results"]["z_vel_abs"][:]
        z_vel_mean_sq = f["Results"]["z_vel_mean_sq"][:]
        bead_patterns = f["Variation"]["Beading_pattern"][:]
        sample_mat = f["Variation"]["Phy_para"][:]
    return z_abs_velocity, z_vel_mean_sq, bead_patterns, sample_mat


def reduce_resolution(z_abs_velocity, resolution=(40, 60)):
    z_abs_velocity = F.interpolate(torch.from_numpy(z_abs_velocity), resolution,  align_corners=True, mode="bilinear")
    z_abs_velocity = z_abs_velocity.numpy().astype(np.float32)
    return z_abs_velocity


def save_data(path, z_abs_velocity, z_vel_mean_sq, bead_patterns, sample_mat):
    assert not os.path.exists(path)
    with h5py.File(path, 'w') as f:
        f.create_dataset("bead_patterns", data=bead_patterns, **compression)
        f.create_dataset("z_vel_mean_sq", data=z_vel_mean_sq, **compression)
        f.create_dataset("z_abs_velocity", data=z_abs_velocity, **compression)
        f.create_dataset("sample_mat", data=sample_mat, **compression)


def add_moments_to_hdf5(path):
    import torch
    eps=1e-12
    with h5py.File(path, 'r+') as f:
        data = torch.from_numpy(f["z_abs_velocity"][:])
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
    z_abs_velocity_list = []
    sample_mat_list = []

    # Read each file and append data to lists
    for file_path in file_list:
        with h5py.File(file_path, 'r') as f:
            bead_patterns_list.append(f["bead_patterns"][:])
            z_vel_mean_sq_list.append(f["z_vel_mean_sq"][:])
            z_abs_velocity_list.append(f["z_abs_velocity"][:])
            sample_mat_list.append(f["sample_mat"][:])

    # Concatenate along the first axis
    bead_patterns_concat = np.concatenate(bead_patterns_list, axis=0)
    z_vel_mean_sq_concat = np.concatenate(z_vel_mean_sq_list, axis=0)
    z_abs_velocity_concat = np.concatenate(z_abs_velocity_list, axis=0)
    sample_mat_concat = np.concatenate(sample_mat_list, axis=0)

    # Save concatenated data to new h5 file
    with h5py.File(output_file, 'w') as f:
        f.create_dataset("bead_patterns", data=bead_patterns_concat, **compression)
        f.create_dataset("z_vel_mean_sq", data=z_vel_mean_sq_concat, **compression)
        f.create_dataset("z_abs_velocity", data=z_abs_velocity_concat, **compression)
        f.create_dataset("sample_mat", data=sample_mat_concat, **compression)


def eval_file(path):
    with h5py.File(path, 'r') as f:
        print(f["bead_patterns"])
        print(f["z_vel_mean_sq"])
        print(f["z_abs_velocity"])
        print(f["sample_mat"])
