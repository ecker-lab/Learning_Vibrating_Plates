import numpy as np
from torch import nn
import torch
from acousticnn.utils.logger import init_train_logger, print_log

import matplotlib.pyplot as plt
from acousticnn.plate.metrics import peak_frequency_error
import os, wandb, scipy, torchvision
from sklearn.neighbors import NearestNeighbors
img_shape = (32, 48)


class AutoEncoder(nn.Module):
    def __init__(self, shape=None):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),  # B, 32, 48 => B, 16, 24
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding="same"),  
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=2, padding=1), # B, 16, 24 => B, 8, 12
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), # B, 8, 12 => B, 4, 6
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1), # B, 4, 6 => B, 2, 3
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1), # B, 5, 8 => B, 10, 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1), # B, 5, 8 => B, 10, 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=1), # B, 10, 15 => B, 20, 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1), # V, 20, 30 => B, 40, 64
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding="same"),  # B, 40, 60=> B, 40, 60
            nn.ReLU(),
            nn.Conv2d(64, 1, 1, stride=1, padding="same"),  # B, 40, 60=> B, 40, 60
        )

    def forward(self, x):
        assert x.shape[-2:] == img_shape # input shape is False
        x = self.encoder(x)
        x = self.decoder(x)
        assert x.shape[-2:] == img_shape # output shape is False
        return x


def train_autoencoder(args, net, dataloader, valloader, optimizer, scheduler, logger=None, lowest=np.inf):
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for batch in dataloader:
            optimizer.zero_grad()
            image, output = batch["bead_patterns"], batch["z_vel_mean_sq"]
            image = nn.functional.interpolate(image, img_shape) /image.max()
            image, output = image.to(args.device), output.to(args.device)
            prediction = net(image)
            loss = torch.nn.functional.mse_loss(prediction, image)
            losses.append(loss.detach().cpu().item())
            loss.backward()
            optimizer.step()
            if scheduler is not None: 
                scheduler.step(epoch)
        print_log(f"Epoch: {epoch}, Training loss: {np.mean(losses):4.4}", logger=logger)
        
        net.eval()
        with torch.no_grad():
            for batch in dataloader:
                image, output = batch["bead_patterns"], batch["z_vel_mean_sq"]
                image = nn.functional.interpolate(image, img_shape) /image.max()
                image, output = image.to(args.device), output.to(args.device)
                prediction = net(image)
                loss_val = torch.nn.functional.mse_loss(prediction, image)
            print_log(f"validation: {loss_val.detach().cpu().numpy():4.4}", logger=logger)
            if loss_val < lowest:
                lowest = loss_val
                if epoch > 20:
                    torch.save(net.state_dict(), os.path.join(args.dir, "checkpoint_best"))

    net.load_state_dict(torch.load(os.path.join(args.dir, "checkpoint_best")))
    return net


def generate_encoding(dataloader, net, use_net=True):
    with torch.no_grad():
        samples = []
        for batch in dataloader:
            image, output, conditional = batch["bead_patterns"], batch["z_vel_mean_sq"], batch["sample_mat"]
            B = image.shape[0]
            if use_net is True:
                image = (nn.functional.interpolate(image, img_shape) /image.max()).cuda()
                encoding = np.hstack((net.encoder(image).view(B, -1).detach().cpu().numpy(), conditional))
            else:
                encoding = np.hstack((image.view(B, -1), conditional))
            samples.append(encoding)
    samples = np.vstack(samples)
    samples = samples.reshape(samples.shape[0], -1)
    return samples


def get_checker(n_neighbors, reference):
    checker = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    checker.fit(reference)
    return checker


def get_predictions(checker, queries, dataset, n_neighbors):
    dist, idx = checker.kneighbors(queries)
    predictions = np.empty((idx.shape[0], n_neighbors, 300))
    for j, i in enumerate(idx):
        predictions[j] = dataset[i.tolist()]["z_vel_mean_sq"]
    return predictions


def generate_plots(checker, dataset, query, query_image):
    dist, idx = checker.kneighbors(query)
    fig, ax = plt.subplots(1, len(idx[0])+1, figsize=(10 / 2.54*len(idx[0]+1), 8 / 2.54))
    ax[0].imshow(query_image, cmap=plt.cm.gray)
    ax[0].axis('off')
    for i, id_ in enumerate(idx[0]):
        ax[i+1].imshow(dataset[id_]["bead_patterns"][0], cmap=plt.cm.gray)
        ax[i+1].axis('off')


def pred_fn(k, trainset, trainloader, dataloader, net, config, use_net=True):
    reference, queries = generate_encoding(trainloader, net, use_net=use_net), generate_encoding(dataloader, net, use_net=use_net)
    checker = get_checker(k, reference)
    prediction = get_predictions(checker, queries, trainset, k)
    prediction = torch.mean(torch.tensor(prediction), dim=1)
    if config.max_frequency is not None:
        prediction = prediction[:, :config.max_frequency]
    return prediction


def get_output(dataset, config):
    output =  dataset[np.arange(len(dataset)).tolist()]["z_vel_mean_sq"]
    if config.max_frequency is not None:
        output = output[:, :config.max_frequency]
    return output


def get_pred_img(k, trainset, trainloader, dataloader, net, use_net=True):
    reference, queries = generate_encoding(trainloader, net, use_net=use_net), generate_encoding(dataloader, net, use_net=use_net)
    checker = get_checker(k, reference)
    dist, idx = checker.kneighbors(queries)
    predictions = np.empty((idx.shape[0], 3,1,81,121))
    for j, i in enumerate(idx):
        predictions[j]  = trainset[i.tolist()]["bead_patterns"]
    return predictions


def eval_knn(reference, queries, k_max, config, logger=None, query_set=None, reference_set=None):
    losses = []
    output = query_set[np.arange(len(query_set)).tolist()]["z_vel_mean_sq"]
    range_vals = (0, k_max)
    for i in range(*range_vals):
        n_neighbors = i+1
        checker = get_checker(n_neighbors, reference)

        prediction = get_predictions(checker, queries, reference_set, n_neighbors)
        prediction = torch.mean(torch.tensor(prediction), dim=1)
        # get actual output
        if config.max_frequency is not None:
            prediction = prediction[:, :config.max_frequency]
            output = output[:, :config.max_frequency]
        losses_per_f = torch.nn.functional.mse_loss(prediction, output, reduction="none").numpy()
        loss = np.mean((losses_per_f))
        losses.append(loss)
        print_log(f"k, {i+1}, Validation loss = {loss:4.2}", logger=logger)
    return losses