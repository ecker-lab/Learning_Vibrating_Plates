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


class npzDataset(Dataset):
    def __init__(self, args, config, data_paths, normalization=True, change_db=False):
        datasets_list = data_paths
        self.images = []
        self.responses = []

        for i, data_path in enumerate(datasets_list):
            data = np.load(data_path)
            images, outputs = np.load(data_path)["images"], np.load(data_path)["responses"]
            if normalization is True:
                if i == 0:
                    if config.data_path_ref is not None:
                        ref_images, ref_outputs = self.calculate_stats(config.data_path_ref)
                    else:
                        ref_images, ref_outputs = images, outputs
                images, outputs = self.normalize(images, outputs, ref_images, ref_outputs)
            if config.cut_frequency is not None:
                print(outputs.shape)
                outputs = outputs[:, :config.cut_frequency]
            self.images.append(images), self.responses.append(outputs)
        self.data = {}
        self.data["bead_patterns"] = torch.tensor(np.vstack(self.images)).float()
        self.data["z_vel_mean_sq"] = torch.tensor(np.vstack(self.responses)).float()
        self.data["sample_mat"] = torch.empty((len(self)))

        #self.data["z_abs_velocity"] = torch.sparse_coo_tensor(size=(len(self), 300, 40, 60))  # using torch sparse to avoid memory usage

        self.keys = ["bead_patterns", "z_vel_mean_sq"]
        del self.images
        del self.responses

        if "sample_mat" not in self.keys:
            print("dummy_sample_mat")
            self.data["sample_mat"] = torch.tensor([0.9, 0.6, 0.003, 0.02]).float().unsqueeze(0).repeat(len(self), 1)
            self.keys.append("sample_mat")

        self.data["sample_mat"] = (self.data["sample_mat"] - MEANS_scalar_parameters) / STD_scalar_parameters

    def calculate_stats(self, data_path_ref):
        print("normalize with: ", data_path_ref)
        if data_path_ref[-3:] == "npz":
            data = np.load(data_path_ref)
            ref_images, ref_outputs = data["images"], data["responses"]
        elif data_path_ref[-2:] == "h5":
            with h5py.File(data_path_ref, 'r') as f:
                ref_images, ref_outputs = f["bead_patterns"][:], f["z_vel_mean_sq"][:]
        else:
            print(data_path_ref)
            raise NotImplementedError
        return ref_images, ref_outputs

    def normalize(self, images, outputs, reference_images, reference_outputs, dont_use_image=True):
        images_mean, images_std = np.mean(reference_images), np.std(reference_images)
        if not dont_use_image:
            images = (images - images_mean) / images_std
        else:
            images = images / 255
        out_mean, out_std = np.mean(reference_outputs, axis=0), np.std((reference_outputs - np.mean(reference_outputs, axis=0)))
        outputs = (outputs - out_mean) / out_std
        self.images_mean, self.images_std = images_mean, images_std
        self.out_mean, self.out_std = out_mean, out_std

        return images, outputs

    def __len__(self):
        return len(self.data["bead_patterns"])

    def __getitem__(self, idx):
        return {key: self.data[key][idx].float() for key in self.keys}


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
            self.normalize_frequency_response(os.path.join(data_dir, config.data_path_ref))
            if "z_abs_velocity" in self.keys:
                self.normalize_field_solution(os.path.join(data_dir, config.data_path_ref))

        if "sample_mat" not in self.keys:
            self.create_dummy_sample_mat()
        if config.filter_dataset is True: 
            self.filter_dataset(config.filter_type, config.filter_orientation)
        self.files["sample_mat"] = (self.files["sample_mat"] - MEANS_scalar_parameters) / STD_scalar_parameters
        self.files["bead_patterns"] = self.files["bead_patterns"].unsqueeze(1)

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

    def normalize_field_solution(self, data_path_ref, normalize=True, eps=1e-12):
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
        self.files["z_abs_velocity"] = torch.log(torch.square((self.files["z_abs_velocity"])) + eps)

        if normalize is True:
            self.files["z_abs_velocity"] = self.files["z_abs_velocity"].sub(self.field_mean).div(self.field_std)

    def normalize_frequency_response(self, data_path_ref):
        with h5py.File(data_path_ref, 'r') as f:
            data = torch.from_numpy(f["z_vel_mean_sq"][:])
        self.out_mean = torch.mean(data, dim=0).float()
        self.out_std = torch.std((data - self.out_mean)).float()
        self.files["z_vel_mean_sq"] = self.files["z_vel_mean_sq"].sub(self.out_mean).div(self.out_std)

    def __len__(self):
        return len(self.files["bead_patterns"])

    def __getitem__(self, index):
        data = {key: self.files[key][index].float() for key in self.keys}
        return data

        
def get_dataloader(args, config, logger, num_workers=1, shuffle=True):
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
        dataset = dataset_class(args, config, config.data_paths)
        trainset, valset = torch.utils.data.random_split(dataset, config.split_ratio)
    else:
        trainset = dataset_class(args, config, config.data_path_train)
        valset = dataset_class(args, config, config.data_path_val)
    if args.wildcard is not None:
        train_percent = args.wildcard/100
        trainset, _ = torch.utils.data.random_split(trainset, (train_percent, 1-train_percent))
        print_log(f"split_trainset{train_percent} len, {len(trainset)}", logger=logger)
    if config.data_paths_test is not None:
        testset = dataset_class(args, config, config.data_paths_test)
    else:
        print_log("use valset for testing", logger=logger)
        testset = valset

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=shuffle, shuffle=shuffle, num_workers=num_workers, pin_memory=True, generator=generator)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers)
    return trainloader, valloader, testloader, trainset, valset, testset
