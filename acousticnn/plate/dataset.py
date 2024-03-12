import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from acousticnn.utils.logger import print_log
import h5py
import hdf5plugin
import os
import torch.nn.functional as F
from acousticnn.plate.configs.main_dir import data_dir

# order of scalar parameters : [l_x, l_y, t, eta]
MEANS_scalar_parameters = torch.tensor([0.75, 0.5, 0.003, 0.02]).float().unsqueeze(0)
STD_scalar_parameters = (torch.tensor([0.3, 0.2, 0.002, 0.02]) / torch.sqrt(torch.tensor(12.0))).float().unsqueeze(0)


class HDF5Dataset(Dataset):
    def __init__(self, args, config, data_paths, normalization=True):
        self.data_paths = [os.path.join(data_dir, data_path) for data_path in data_paths]
        self.keys = config.dataset_keys
        self.files = [h5py.File(path, 'r') for path in self.data_paths]
        self.normalization = normalization

        self.files = [{key: torch.from_numpy(f[key][:]) for key in self.keys} for f in self.files]
        self.files = {key: torch.cat([d[key] for d in self.files], dim=0) for key in self.keys}

        if self.normalization is True:
            print("normalize with: ", config.data_path_ref)
            self.get_moments(os.path.join(data_dir, config.data_path_ref))
            self.normalize_frequency_response()
            if "z_abs_velocity" in self.keys:
                self.normalize_field_solution()

        if config.filter_dataset is True: 
            self.filter_dataset(config.filter_type, config.filter_orientation)
        if "sample_mat" in self.keys and config.conditional:
            self.files["sample_mat"] = (self.files["sample_mat"] - torch.tensor(config.mean_conditional_param).float().unsqueeze(0))\
                                                                                         / torch.tensor(config.std_conditional_param).float().unsqueeze(0)
        if "sample_mat" not in self.keys:
            self.create_dummy_sample_mat()

        self.files["bead_patterns"] = self.files["bead_patterns"].unsqueeze(1) / self.files["bead_patterns"].max()
        if config.n_frequencies < 300:
            print(self.files["z_abs_velocity"].shape, self.files["z_vel_mean_sq"].shape, self.field_mean.shape)
            self.files["z_abs_velocity"] = self.files["z_abs_velocity"][:,:config.n_frequencies]
            self.files["z_vel_mean_sq"] = self.files["z_vel_mean_sq"][:,:config.n_frequencies]
            self.field_mean = self.field_mean[:, :config.n_frequencies]
            self.out_mean = self.out_mean[:config.n_frequencies]
            print(self.files["z_abs_velocity"].shape, self.files["z_vel_mean_sq"].shape, self.field_mean.shape)
            print(self.out_mean.shape)

        if hasattr(config, "rmfreqs"):
            if config.rmfreqs is True:
                self.files["z_abs_velocity"] = torch.from_numpy(np.concatenate((self.files["z_abs_velocity"][:, :50], self.files["z_abs_velocity"][:, 100:]), axis=1))
                self.files["z_vel_mean_sq"] = torch.from_numpy(np.concatenate((self.files["z_vel_mean_sq"][:, :50], self.files["z_vel_mean_sq"][:, 100:]), axis=1))
                self.field_mean = torch.from_numpy(np.concatenate((self.field_mean[:, :50], self.field_mean[:, 100:]), axis=1))
                self.out_mean = torch.from_numpy(np.concatenate((self.out_mean[:50], self.out_mean[100:]), axis=0))

    def create_dummy_sample_mat(self):
        print("dummy_sample_mat")
        self.files["sample_mat"] = torch.tensor([0.9, 0.6, 0.003, 0.02]).float().unsqueeze(0).repeat(len(self), 1)
        self.keys.append("sample_mat")

    def filter_dataset(self, filter_type, filter_orientation):
        if filter_type == "thickness":
            if filter_orientation == "larger":
                mask = self.files["sample_mat"][:, 2] > 0.02
            elif filter_orientation == "smaller":
                mask = self.files["sample_mat"][:, 2] < 0.02
        elif filter_type == "bead_ratio":
            bead_ratio = torch.sum(self.files["bead_patterns"] > 0, dim=(1, 2)) / (self.files["bead_patterns"].shape[1] * self.files["bead_patterns"].shape[2])
            avg_ratio = torch.quantile(bead_ratio, 0.5)
            if filter_orientation == "larger":
                mask = bead_ratio > avg_ratio
            elif filter_orientation == "smaller":
                mask = bead_ratio < avg_ratio
        for key in self.keys:
            self.files[key] = self.files[key][mask]
        print(len(self))

    def normalize_field_solution(self, normalize=True, eps=1e-12):
        self.files["z_abs_velocity"] = torch.log(torch.square((self.files["z_abs_velocity"])) + eps)
        if normalize is True:
            self.files["z_abs_velocity"] = self.files["z_abs_velocity"].sub(self.field_mean).div(self.field_std)

    def normalize_frequency_response(self):
        self.files["z_vel_mean_sq"] = self.files["z_vel_mean_sq"].sub(self.out_mean).div(self.out_std)

    def __len__(self):
        return len(self.files["bead_patterns"])

    def __getitem__(self, index):
        data = {key: self.files[key][index].float() for key in self.keys}
        return data

    def get_moments(self, data_path_ref, eps=1e-12):
        with h5py.File(data_path_ref, 'r') as f:
            if "field_mean" not in f.keys():
                data = torch.from_numpy(f["z_abs_velocity"][:])
                data = torch.log(torch.square(data) + eps)
                self.field_mean = torch.mean(data, axis=(0, 2, 3)).view(1, -1, 1, 1)
                self.field_std = torch.std(data - self.field_mean) # here the - self.field_mean is optional
            else: 
                print("use precomputed moments")
                self.field_mean = torch.from_numpy(f["field_mean"][:])
                self.field_std = torch.tensor(f["field_std"][()])
            data = torch.from_numpy(f["z_vel_mean_sq"][:])
        self.out_mean = torch.mean(data, dim=0).float()
        self.out_std = torch.std((data - self.out_mean)).float()

    def undo_field_transformation(self, vel_field, eps=1e-12):
        return torch.sqrt(torch.exp(vel_field))

            

def get_dataloader(args, config, logger, num_workers=1, shuffle=True, normalization=True):
    batch_size = args.batch_size
    dataset_class = config.dataset_class
    if dataset_class == "npz":
        dataset_class = npzDataset
    elif dataset_class == "hdf5":
        dataset_class = HDF5Dataset
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.FloatTensor')
    generator = torch.Generator(device='cpu')

    if config.random_split is True:
        dataset = dataset_class(args, config, config.data_paths, normalization=normalization)
        trainset, valset = torch.utils.data.random_split(dataset, config.split_ratio)
    else:
        trainset = dataset_class(args, config, config.data_path_train, normalization=normalization)
        valset = dataset_class(args, config, config.data_path_val, normalization=normalization)
    if args.wildcard is not None:
        train_percent = args.wildcard/100
        trainset, _ = torch.utils.data.random_split(trainset, (train_percent, 1-train_percent))
        print_log(f"split_trainset{train_percent} len, {len(trainset)}", logger=logger)
    if config.data_paths_test is not None:
        testset = dataset_class(args, config, config.data_paths_test, normalization=normalization)
    else:
        print_log("use valset for testing", logger=logger)
        testset = valset

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=shuffle, shuffle=shuffle, num_workers=num_workers, pin_memory=True, generator=generator)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers)
    return trainloader, valloader, testloader, trainset, valset, testset
