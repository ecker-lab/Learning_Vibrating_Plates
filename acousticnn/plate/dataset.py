import torch
from torch.utils.data import Dataset
import numpy as np
from acousticnn.utils.logger import print_log
import h5py
import hdf5plugin
import os
from acousticnn.main_dir import data_dir
from torchinterp1d import interp1d
import csv


class HDF5Dataset(Dataset):
    def __init__(self, args, config, data_paths, test=True, normalization=True, sample_idx=slice(None), freq_idx=None):
        self.data_paths = [os.path.join(data_dir, data_path) for data_path in data_paths]
        self.keys = set(config.dataset_keys.copy())
        self.files = [(h5py.File(path, 'r'), path) for path in self.data_paths]
        self.normalization = normalization
        self.freq_sampling = test == False and config.freq_sample == True
        self.freq_sampling_limit = config.freq_limit if hasattr(config, "freq_limit") else 300
        self.config = config
        self.files_loading = [{key: torch.from_numpy(f[key][:]).float() for key in self.keys} for f, path in self.files]

        # generate freq_idx filter for training
        if hasattr(config, "freq_limit") and freq_idx is None and test==False:
            n_samples, n_freqs = self.files["z_vel_mean_sq"].shape[:2]
            freq_idx = torch.stack([torch.randperm(n_freqs)[:config.freq_limit] for _ in range(n_samples)]).numpy()

        # only select samples specified by sample idx and freq idx
        if freq_idx is None:
            self.files = {key: torch.cat([d[key] for d in self.files_loading], dim=0)[sample_idx] for key in self.keys}
        else:
            self.files = {}
            for key in self.keys & {"bead_patterns", "phy_para"}:
                self.files[key] = torch.cat([d[key] for d in self.files_loading], dim=0)[sample_idx]
            sample_idx = np.repeat(sample_idx.reshape(-1, 1), freq_idx.shape[1], axis=1)
            for key in self.keys - {"bead_patterns", "phy_para"}:
                self.files[key] = torch.cat([d[key] for d in self.files_loading], dim=0)[sample_idx, freq_idx]
            del self.files_loading

        if self.normalization is True:
            self.files["bead_patterns"] = self.files["bead_patterns"].unsqueeze(1) / self.files["bead_patterns"].max()
            print("normalize with: ", config.data_path_ref)
            self.get_moments(os.path.join(data_dir, config.data_path_ref))
            self.normalize_frequencies()
            self.normalize_frequency_response()
            if "z_vel_abs" in self.keys:
                self.normalize_field_solution()

        self.handle_conditional_params(config)

    def handle_conditional_params(self, config):
        if "phy_para" not in self.keys:
            print("dummy_phy_para")
            self.files["phy_para"] = torch.tensor([0.9, 0.6, 0.003, 0.02, 0, 0.36, 0.225]).float().unsqueeze(0).repeat(len(self), 1)
            self.keys.add("phy_para")
        self.files["phy_para"] = (self.files["phy_para"] - torch.tensor(config.mean_conditional_param).float().unsqueeze(0))\
        / torch.tensor(config.std_conditional_param).float().unsqueeze(0)

    def get_moments(self, data_path_ref, eps=1e-12):
        with h5py.File(data_path_ref, 'r') as f:
            if "field_mean" not in f.keys():
                data = torch.from_numpy(f["z_vel_abs"][:])
                data = torch.log(torch.square(data) + eps)
                self.field_mean = torch.mean(data, axis=(0, 2, 3))
                self.field_std = torch.std(data - self.field_mean)
            else:
                print("use precomputed moments")
                self.field_mean = torch.from_numpy(f["field_mean"][:])
                self.field_std = torch.tensor(f["field_std"][()])

            self.field_mean = convert_1d_to_interpolator(self.field_mean.flatten(), -1, 1)
            data = torch.from_numpy(f["z_vel_mean_sq"][:])
        self.out_mean = torch.mean(data, dim=0).float()
        self.out_std = torch.std((data - self.out_mean)).float()
        self.out_mean = convert_1d_to_interpolator(self.out_mean, -1, 1)

    def normalize_frequencies(self):
        self.files["frequencies"] = (self.files["frequencies"] - 1) / 299 * 2 - 1

    def normalize_frequency_response(self):
        self.files["z_vel_mean_sq"] = self.files["z_vel_mean_sq"].sub(self.out_mean[self.files['frequencies']]).div(self.out_std)

    def normalize_field_solution(self, normalize=True, eps=1e-12):
        self.files["z_vel_abs"] = torch.log(torch.square((self.files["z_vel_abs"])) + eps)
        if normalize is True:
            self.files["z_vel_abs"] = self.files["z_vel_abs"].sub(self.field_mean[self.files['frequencies']].unsqueeze(-1).unsqueeze(-1)).div(self.field_std)

    def undo_field_transformation(self, vel_field, eps=1e-12):
        return torch.sqrt(torch.exp(vel_field))

    def filter_dataset(self, filter_orientation, filter_type='bead_ratio'):
        if filter_type == "bead_ratio":
            bead_ratio = torch.sum(self.files["bead_patterns"] > 0, dim=(1, 2, 3)) / (self.files["bead_patterns"].shape[2] * self.files["bead_patterns"].shape[3])
            avg_ratio = torch.quantile(bead_ratio, 0.5)
            if filter_orientation == "larger":
                mask = bead_ratio > avg_ratio
            elif filter_orientation == "smaller":
                mask = bead_ratio <= avg_ratio
            else:
                raise ValueError("Invalid filter orientation", filter_orientation)
        for key in self.keys:
            self.files[key] = self.files[key][mask]
        print(len(self))

    def __len__(self):
        return len(self.files["z_vel_mean_sq"])

    def __getitem__(self, index):
        data = {key: self.files[key][index] for key in self.keys}
        if self.freq_sampling is True:
            freq_sampling_idx = torch.randint(self.freq_sampling_limit, (self.freq_sampling_limit,))
            data['z_vel_mean_sq'] = data['z_vel_mean_sq'][freq_sampling_idx]
            if 'z_vel_abs' in self.keys:
                data['z_vel_abs'] = data['z_vel_abs'][freq_sampling_idx]
            data['frequencies'] = data['frequencies'][freq_sampling_idx]

        return data


class convert_1d_to_interpolator:
    def __init__(self, array, min_val, max_val):
        self.array = array
        self.min_val = min_val
        self.max_val = max_val
        self.x = torch.linspace(min_val, max_val, steps=array.shape[0], device=array.device)

    def __getitem__(self, xnew):
        if not isinstance(xnew, torch.Tensor):
            xnew = torch.tensor(xnew, dtype=torch.float32, device=self.array.device)
        original_shape = xnew.shape
        xnew_flat = xnew.flatten()
        interpolated_values_flat = interp1d(self.x, self.array, xnew_flat, None)
        return interpolated_values_flat.view(original_shape)

    def cuda(self):
        self.array = self.array.cuda()
        self.x = self.x.cuda()
        return self

    def to(self, device, dtype=None):
        self.array = self.array.to(device, dtype=dtype)
        self.x = self.x.to(device, dtype=dtype)
        return self


def get_dataloader(args, config, logger, num_workers=0, shuffle=True, normalization=True):
    batch_size = args.batch_size
    np.random.seed(args.seed), torch.cuda.manual_seed_all(args.seed), torch.manual_seed(args.seed)
    generator = torch.Generator(device='cpu')
    if hasattr(args, "wildcard") and args.wildcard is not None:
        config.n_train_samples = int(np.floor(args.wildcard*config.n_train_samples/100))
    idx = torch.randperm(config.n_samples)[:config.n_train_samples].numpy()
    if hasattr(config, "freq_limit"):
        freq_idx = torch.stack([torch.randperm(config.n_freqs)[:config.freq_limit] for _ in range(config.n_train_samples)]).numpy()
    else:
        freq_idx = None
    if config.random_split is False:
        trainset = HDF5Dataset(args, config, config.data_path_train, normalization=normalization, test=False, \
                            sample_idx=idx, freq_idx=freq_idx)
        valset = HDF5Dataset(args, config, config.data_path_val, normalization=normalization, \
                            sample_idx=torch.randperm(1000)[:config.n_val_samples], test=True) # TODO remove hardcoding
    else:
        idx = torch.randperm(config.n_samples).numpy()
        trainset = HDF5Dataset(args, config, config.data_path_train, normalization=normalization, test=False, \
                            sample_idx=idx[:config.n_train_samples], freq_idx=freq_idx)
        valset = HDF5Dataset(args, config, config.data_path_train, normalization=normalization, \
                            sample_idx=idx[-config.n_val_samples:], test=True)
    if config.data_paths_test is not None:
        testset = HDF5Dataset(args, config, config.data_paths_test, normalization=normalization, test=True)
    else:
        print_log("use valset for testing", logger=logger)
        testset = valset
    if args.filter_dataset != 'False':
        trainset.filter_dataset(args.filter_dataset)
        valset.filter_dataset(args.filter_dataset)
        testset.filter_dataset(args.filter_dataset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=shuffle, shuffle=shuffle, num_workers=num_workers, pin_memory=True, generator=generator)
    valloader = torch.utils.data.DataLoader(valset, batch_size=2, drop_last=False, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2, drop_last=False, shuffle=False, num_workers=num_workers)
    return trainloader, valloader, testloader, trainset, valset, testset