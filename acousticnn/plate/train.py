import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from acousticnn.utils.logger import print_log
from acousticnn.plate.metrics import peak_frequency_error, compute_wasserstein_distance
import wandb


def train(args, config, net, dataloader, optimizer, valloader, scheduler, logger=None):
    lowest = np.inf
    net.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    for epoch in range(config.epochs):
        losses = []
        for batch in dataloader:
            optimizer.zero_grad()
            image, output, condition = batch["bead_patterns"], batch["z_vel_mean_sq"], batch["sample_mat"]
            image, output, condition = image.to(args.device), output.to(args.device), condition.to(args.device)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                prediction = net(image, condition)
                #loss = torch.nn.functional.mse_loss(prediction, output)
                loss = torch.nn.functional.l1_loss(prediction, output)
            losses.append(loss.detach().cpu().item())
            scaler.scale(loss.mean()).backward()
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step(epoch)

        print_log(f"Epoch {epoch} training loss = {(np.mean(losses)):4.4}", logger=logger)
        if logger is not None:
            wandb.log({'Loss Freq / Training': np.mean(losses), 'LR': optimizer.param_groups[0]['lr'], 'Epoch': epoch})

        if epoch % config.validation_frequency == 0 or epoch % int(config.epochs/10) == 0:
            save_model(args.dir, epoch, net, optimizer, loss, "checkpoint_last")
            loss = evaluate(args, config, net, valloader, logger=logger, epoch=epoch)["loss (test/val)"]
            if loss < lowest:
                print_log("best model", logger=logger)
                save_model(args.dir, epoch, net, optimizer, loss)
                lowest = loss
        if epoch == (config.epochs - 1):
            path = os.path.join(args.dir, "checkpoint_best")
            net.load_state_dict(torch.load(path)["model_state_dict"])
            _ = evaluate(args, config, net, valloader, logger, True, report_peak_error=True, epoch=epoch, report_wasserstein=True)
    return net


def evaluate(args, config, net, dataloader, logger=None, plot=False, report_peak_error=True, epoch=None, report_wasserstein=True, verbose=True):
    prediction, output = _generate_preds(args, config, net, dataloader)
    results = _evaluate(prediction, output, logger, config, args, epoch, report_peak_error, report_wasserstein, dataloader, verbose)
    return results


def _generate_preds(args, config, net, dataloader):
    net.eval()
    with torch.no_grad():
        predictions, outputs = [], []
        for batch in dataloader:
            image, output, condition = batch["bead_patterns"], batch["z_vel_mean_sq"], batch["sample_mat"]
            image, output, condition = image.to(args.device), output.to(args.device), condition.to(args.device)
            prediction = net(image, condition)
            if config.max_frequency is not None:
                prediction, output = prediction[:, :config.max_frequency], output[:, :config.max_frequency]
            predictions.append(prediction.detach().cpu()), outputs.append(output.detach().cpu())
    return torch.vstack(predictions), torch.vstack(outputs)


def _evaluate(prediction, output, logger, config, args, epoch, report_peak_error, report_wasserstein, dataloader, verbose=True, field_losses=None):
    REPORT_L1_LOSS = True
    results = {}
    losses_per_f = torch.nn.functional.mse_loss(prediction, output, reduction="none")
    prediction, output, losses_per_f = prediction.numpy(), output.numpy(), losses_per_f.numpy()
    loss = np.mean(losses_per_f)
    results.update({"losses_per_f": losses_per_f, "loss (test/val)": loss})
    mean, std = extract_mean_std(dataloader.dataset)
    if report_peak_error is True:
        results_peak = peak_frequency_error(output, prediction)
        ad, fd = np.nanmean(results_peak["amplitude_distance"]), np.nanmean(results_peak["frequency_distance"])
        save_rmean = 1 - np.nanmean(results_peak["save_peak_ratio"])
        results.update({"peak_ratio": results_peak["peak_ratio"], "amplitude_distance": ad, "frequency_distance": fd, "save_rmean": save_rmean})
    if report_wasserstein is True:
        wasserstein = compute_wasserstein_distance(output, prediction, mean, std)
        results.update({"wasserstein": wasserstein})
    if field_losses is not None:
        results.update(field_losses) 
    if REPORT_L1_LOSS is True:
        results.update({"L1 Loss / (test/val)": np.mean(np.abs(prediction - output))})
    for key in results.keys():
        if key == "losses_per_f" or key == "peak_ratio":
            continue
        if verbose is True:
            print_log(f"{key} = {results[key]:4.4}", logger=logger)
        if logger is not None:
            wandb.log({key: results[key], 'Epoch': epoch})

    return results


def extract_mean_std(dataset):
    def get_base_dataset(dataset):
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        return dataset

    base_dataset = get_base_dataset(dataset)
    mean, std = base_dataset.out_mean, base_dataset.out_std

    #return mean, std
    try:
        mean, std = mean.numpy(), std.numpy()
    except: 
        pass
    return mean, std

def save_model(savepath, epoch, model, optimizer, loss, name="checkpoint_best"):
    torch.save({
               'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'loss': loss
               },
               os.path.join(savepath, name))

