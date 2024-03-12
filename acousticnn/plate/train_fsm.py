import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from acousticnn.utils.logger import print_log
from acousticnn.plate.metrics import peak_frequency_error, compute_wasserstein_distance
from acousticnn.plate.train import _evaluate, save_model
import wandb


def add_noise(image, add_noise=False):
    if add_noise:
        #noise_levels = (torch.rand(image.size(0), 1, 1, 1) - .5) * 0.8 * image.max()
        #noisy_image = image + torch.randn_like(image) * noise_levels
        noise_levels = (torch.rand(image.size(0), 1, 1, 1) - .5).to(image.device) * 1 * image.max()
        noisy_image = image + torch.randn_like(image) * noise_levels
        noisy_image.clamp_(image.min(), image.max())
    else:
        noisy_image = image
    return noisy_image


def train(args, config, net, dataloader, optimizer, valloader, scheduler, logger=None, start_epoch=0, lowest_loss=np.inf):
    lowest = lowest_loss
    net.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    out_mean, out_std, field_mean, field_std = extract_mean_std(dataloader.dataset)
    out_mean, out_std = out_mean.to(args.device), out_std.to(args.device)
    field_mean, field_std = field_mean.to(args.device), field_std.to(args.device)

    for epoch in range(start_epoch + 1, config.epochs):
        losses_freq, losses_field = [], []
        for batch in dataloader:
            optimizer.zero_grad()
            image, field_solution, output, condition = batch["bead_patterns"], batch["z_abs_velocity"], batch["z_vel_mean_sq"], batch["sample_mat"]
            image = add_noise(image, args.add_noise)
            image, field_solution, output, condition = image.to(args.device), field_solution.to(args.device), output.to(args.device), condition.to(args.device)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                prediction_field = net(image, condition)
                loss_field = F.mse_loss(prediction_field, field_solution)
                prediction = get_mean_from_field_solution(prediction_field, field_mean, field_std, config.n_frequencies)
                prediction.sub_(out_mean).div_(out_std)
                loss_freq = F.mse_loss(prediction, output)
                if not torch.isnan(loss_freq).any():
                    loss = args.alpha * loss_field + (1-args.alpha) *loss_freq
                else:
                    print_log("nan", logger=logger)
                    loss = loss_field
            losses_field.append(loss_field.detach().cpu().item())
            losses_freq.append(loss_freq.detach().cpu().item())
            scaler.scale(loss.mean()).backward()
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step(epoch)

        print_log(f"Epoch {epoch} training loss = {(np.mean(losses_field)):4.4}, {(np.mean(losses_freq)):4.4}", logger=logger)
        if logger is not None:
            wandb.log({'Loss Field / Training': np.mean(losses_field), 'Loss Freq / Training': np.mean(losses_freq), 'LR': optimizer.param_groups[0]['lr'], 'Epoch': epoch})

        if epoch % config.validation_frequency == 0:
            save_model(args.dir, epoch, net, optimizer, lowest, "checkpoint_last")
            loss = evaluate(args, config, net, valloader, logger=logger, plot=True, epoch=epoch, report_wasserstein=config.report_wasserstein)["loss (test/val)"]
            if loss < lowest:
                print_log("best model", logger=logger)
                save_model(args.dir, epoch, net, optimizer, lowest)
                lowest = loss
        if epoch == (config.epochs - 1):
            path = os.path.join(args.dir, "checkpoint_best")
            net.load_state_dict(torch.load(path)["model_state_dict"])
            _ = evaluate(args, config, net, valloader, logger=logger, plot=True, report_peak_error=True, epoch=epoch, report_wasserstein=config.report_wasserstein)
    return net


def evaluate(args, config, net, dataloader, logger=None, plot=False, report_peak_error=True, epoch=None, report_wasserstein=False, verbose=True):
    prediction, output, field_losses = _generate_preds(args, config, net, dataloader)
    results = _evaluate(prediction, output, logger, config, args, epoch, report_peak_error, report_wasserstein, dataloader, verbose, field_losses)

    return results


def _generate_preds(args, config, net, dataloader):
    net.eval()
    with torch.no_grad():
        predictions, outputs, mse_losses, l1_losses = [], [], [], []
        out_mean, out_std, field_mean, field_std = extract_mean_std(dataloader.dataset)
        out_mean, out_std, field_mean, field_std = out_mean.to(args.device), out_std.to(args.device), field_mean.to(args.device), field_std.to(args.device)
        for batch in dataloader:
            image, field_solution, output, condition = batch["bead_patterns"], batch["z_abs_velocity"], batch["z_vel_mean_sq"], batch["sample_mat"]
            image, field_solution, output, condition = image.to(args.device), field_solution.to(args.device), output.to(args.device), condition.to(args.device)
            prediction_field = net(image, condition)
            mse_losses.append(F.mse_loss(prediction_field, field_solution).detach().cpu().numpy())
            l1_losses.append(F.l1_loss(prediction_field, field_solution).detach().cpu().numpy())
            prediction = get_mean_from_field_solution(prediction_field, field_mean, field_std, config.n_frequencies)
            prediction.sub_(out_mean).div_(out_std)
            if config.max_frequency is not None:
                prediction, output = prediction[:, :config.max_frequency], output[:, :config.max_frequency]
            predictions.append(prediction.detach().cpu()), outputs.append(output.detach().cpu())
    return torch.vstack(predictions), torch.vstack(outputs), {"Loss Field / (test/val)": np.mean(mse_losses), "L1 Loss Field / (test/val)": np.mean(l1_losses)}


def extract_mean_std(dataset, device="cuda"):
    def get_base_dataset(dataset):
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        return dataset

    base_dataset = get_base_dataset(dataset)
    out_mean, out_std = base_dataset.out_mean, base_dataset.out_std
    field_mean, field_std = base_dataset.field_mean, base_dataset.field_std
    return out_mean, out_std, field_mean, field_std


def get_mean_from_field_solution(field_solution, field_mean, field_std, n_frequencies=300):
    v_ref = 1e-9
    eps = 1e-12
    shape = field_solution.shape
    n_frequencies = field_mean.shape[1]
    # remove transformations in dataloader
    field_solution = field_solution * field_std + field_mean
    field_solution = torch.exp(field_solution) - eps
    # calculate frequency response
    field_solution = field_solution.view(shape[0], n_frequencies, -1)
    v = torch.mean(field_solution, dim=-1)
    v = 10*torch.log10((v + 1e-12)/v_ref)
    return v.view(shape[0], n_frequencies)
