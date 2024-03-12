import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from acousticnn.utils.logger import print_log
from acousticnn.plate.metrics import peak_frequency_error, compute_wasserstein_distance
from acousticnn.plate.train import _evaluate, save_model
import wandb


def train(args, config, net, dataloader, optimizer, valloader, scheduler, logger=None):
    lowest = np.inf
    net.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    out_mean, out_std, field_mean, field_std = extract_mean_std(dataloader.dataset)
    out_mean, out_std = out_mean.to(args.device), out_std.to(args.device)
    field_mean, field_std = field_mean.to(args.device), field_std.to(args.device)
    for epoch in range(config.epochs):
        losses_freq, losses_field = [], []
        for batch in dataloader:
            optimizer.zero_grad()
            image, field_solution, output, condition = batch["bead_patterns"], batch["z_abs_velocity"], batch["z_vel_mean_sq"], batch["sample_mat"]
            image, field_solution, output, condition = image.to(args.device), field_solution.to(args.device), output.to(args.device), condition.to(args.device)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                prediction_field = net(image, condition)
                loss_field = F.mse_loss(prediction_field, field_solution)
                prediction = get_mean_from_field_solution(prediction_field, field_mean, field_std)
                prediction.sub_(out_mean).div_(out_std)
                loss_freq = F.mse_loss(prediction, output)
                if not torch.isnan(loss_freq).any():
                    loss = (loss_field + loss_freq) / 2
                else:
                    print_log("loss", end=", ")
                    #print_log("loss_nan", logger=logger)
                    loss = loss_field
            #print(loss_field.item(), loss_freq.item())
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

        if epoch % config.validation_frequency == 0 or epoch % int(config.epochs/10) == 0:
            save_model(args.dir, epoch, net, optimizer, loss, "checkpoint_last")
            loss = evaluate(args, config, net, valloader, logger=logger, epoch=epoch, report_wasserstein=config.report_wasserstein)["loss (test/val)"]
            if loss < lowest:
                print_log("best model", logger=logger)
                save_model(args.dir, epoch, net, optimizer, loss)
                lowest = loss
        if epoch == (config.epochs - 1):
            path = os.path.join(args.dir, "checkpoint_best")
            net.load_state_dict(torch.load(path)["model_state_dict"])
            _ = evaluate(args, config, net, valloader, logger, True, report_peak_error=True, epoch=epoch, report_wasserstein=config.report_wasserstein)
    return net


def evaluate(args, config, net, dataloader, logger=None, plot=False, report_peak_error=True, epoch=None, report_wasserstein=False, verbose=True):
    prediction, output, losses = _generate_preds(args, config, net, dataloader)
    print_log(f"loss_field (test/val): {np.mean(losses):4.4f}", logger=logger)
    if logger is not None:
        wandb.log({'Loss Field / (test/val)': np.mean(losses)})
    results = _evaluate(prediction, output, logger, config, args, epoch, report_peak_error, report_wasserstein, dataloader, verbose)
    return results


def _generate_preds(args, config, net, dataloader):
    net.eval()
    with torch.no_grad():
        predictions, outputs, losses = [], [], []
        out_mean, out_std, field_mean, field_std = extract_mean_std(dataloader.dataset)
        out_mean, out_std = torch.tensor(out_mean).to(args.device), torch.tensor(out_std).to(args.device)
        field_mean, field_std = torch.tensor(field_mean).to(args.device), torch.tensor(field_std).to(args.device)
        for batch in dataloader:
            image, field_solution, output, condition = batch["bead_patterns"], batch["z_abs_velocity"], batch["z_vel_mean_sq"], batch["sample_mat"]
            image, field_solution, output, condition = image.to(args.device), field_solution.to(args.device), output.to(args.device), condition.to(args.device)
            prediction_field = net(image, condition)
            losses.append(F.mse_loss(prediction_field, field_solution).detach().cpu().numpy())
            prediction = get_mean_from_field_solution(prediction_field, field_mean, field_std)
            prediction.sub_(out_mean).div_(out_std)
            if config.max_frequency is not None:
                prediction, output = prediction[:, :config.max_frequency], output[:, :config.max_frequency]
            predictions.append(prediction.detach().cpu()), outputs.append(output.detach().cpu())
    return torch.vstack(predictions), torch.vstack(outputs), losses


def extract_mean_std(dataset, device="cuda"):
    def get_base_dataset(dataset):
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        return dataset

    base_dataset = get_base_dataset(dataset)
    out_mean, out_std = base_dataset.out_mean, base_dataset.out_std
    field_mean, field_std = base_dataset.field_mean, base_dataset.field_std
    return out_mean, out_std, field_mean, field_std


def get_mean_from_field_solution(field_solution, field_mean, field_std):
    v_ref = 1e-9
    eps = 1e-12
    shape = field_solution.shape
    # remove transformations in dataloader
    field_solution = field_solution * field_std + field_mean
    field_solution = torch.exp(field_solution) - eps
    # calculate frequency response
    field_solution = field_solution.view(shape[0], 300, -1)
    v = torch.mean(field_solution, dim=-1)
    v = 10*torch.log10((v + 1e-12)/v_ref)
    return v.view(shape[0], 300)
