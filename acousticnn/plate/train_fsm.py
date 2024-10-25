import os
import numpy as np
import torch
import torch.nn.functional as F
from acousticnn.utils.logger import print_log
from acousticnn.plate.train import _evaluate, save_model
import wandb


fields = ["bead_patterns", "z_vel_abs", "z_vel_mean_sq", "phy_para", "frequencies"]
wasserstein = False

def add_noise(image, add_noise=False):
    if add_noise:
        noise_levels = (torch.rand(image.size(0), 1, 1, 1) - .5).to(image.device) * image.max()
        noisy_image = image + torch.randn_like(image) * noise_levels
        noisy_image.clamp_(image.min(), image.max())
    else:
        noisy_image = image
    return noisy_image


def train(args, config, model_cfg, net, dataloader, optimizer, valloader, scheduler, logger=None, start_epoch=0, lowest_loss=np.inf):
    net.train()
    if args.compile:
        torch.set_float32_matmul_precision('high')
        net = torch.compile(net)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    out_mean, out_std, field_mean, field_std = extract_mean_std(dataloader.dataset)
    out_mean, out_std = out_mean.to(args.device), out_std.to(args.device)
    field_mean, field_std = field_mean.to(args.device), field_std.to(args.device)

    for epoch in range(start_epoch + 1, config.epochs):
        losses_freq, losses_field = [], []
        for batch in dataloader:
            optimizer.zero_grad()
            image, field_solution, vel_mean_sq, condition, frequencies = (batch[field].to(args.device) for field in fields)
            target = field_solution if model_cfg.velocity_field else vel_mean_sq
            image = add_noise(image, args.add_noise)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                prediction = net(image, condition, frequencies)
                loss = F.mse_loss(prediction, target)
            losses_field.append(loss.detach().cpu().item())
            scaler.scale(loss.mean()).backward()

            if config.optimizer.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.optimizer.gradient_clip)

            scaler.step(optimizer), scaler.update()

            with torch.no_grad():
                if model_cfg.velocity_field:
                    prediction = get_mean_from_field_solution(prediction, field_mean, field_std, frequencies)
                prediction.sub_(out_mean[frequencies]).div_(out_std)
                loss_freq = F.mse_loss(prediction, vel_mean_sq, reduction='none')
                #print_log(f'{loss_freq.detach().cpu().shape}', logger=logger)
                losses_freq.append(loss_freq.detach().cpu().mean())
        if scheduler is not None:
            scheduler.step(epoch)

        print_log(f"Epoch {epoch} training loss = {(np.mean(losses_field)):4.4}, {(np.mean(losses_freq)):4.4}", logger=logger)
        if logger is not None:
            wandb.log({'Loss Field / Training': np.mean(losses_field), 'Loss Freq / Training': np.mean(losses_freq), 'LR': optimizer.param_groups[0]['lr'], 'Epoch': epoch})


        if epoch % config.validation_frequency == 0:
            save_model(args.dir, epoch, net, optimizer, lowest_loss, "checkpoint_last")
            net.eval()
            loss = evaluate(args, config, net, valloader, logger=logger, epoch=epoch, report_wasserstein=wasserstein)["loss (test/val)"]
            if loss < lowest_loss:
                print_log("best model", logger=logger)
                save_model(args.dir, epoch, net, optimizer, lowest_loss)
                lowest_loss = loss
        if epoch == (config.epochs - 1):
            path = os.path.join(args.dir, "checkpoint_best")
            net.load_state_dict(torch.load(path)["model_state_dict"])
            _ = evaluate(args, config, net, valloader, logger=logger, report_peak_error=True, epoch=epoch, report_wasserstein=wasserstein)
    return net


def evaluate(args, config, net, dataloader, logger=None, report_peak_error=True, epoch=None, report_wasserstein=False, verbose=True):
    prediction, output, field_losses = _generate_preds(args, config, net, dataloader)
    results = _evaluate(prediction, output, logger, config, args, epoch, report_peak_error, report_wasserstein, dataloader, verbose, field_losses)
    return results


def _generate_preds(args, config, net, dataloader):
    with torch.no_grad():
        predictions, outputs, mse_losses, l1_losses = [], [], [], []
        out_mean, out_std, field_mean, field_std = extract_mean_std(dataloader.dataset)
        out_mean, out_std, field_mean, field_std = out_mean.to(args.device), out_std.to(args.device), field_mean.to(args.device), field_std.to(args.device)
        for batch in dataloader:
            image, field_solution, vel_mean_sq, condition, frequencies = (batch[field].to(args.device) for field in fields)
            prediction_field = net(image, condition, frequencies)
            mse_losses.append(F.mse_loss(prediction_field, field_solution).detach().cpu().numpy())
            l1_losses.append(F.l1_loss(prediction_field, field_solution).detach().cpu().numpy())
            prediction = get_mean_from_field_solution(prediction_field, field_mean, field_std, frequencies)
            prediction.sub_(out_mean[frequencies]).div_(out_std)
            if config.max_frequency is not None:
                prediction, vel_mean_sq = prediction[:, :config.max_frequency], vel_mean_sq[:, :config.max_frequency]
            predictions.append(prediction.detach().cpu()), outputs.append(vel_mean_sq.detach().cpu())
    return torch.vstack(predictions), torch.vstack(outputs), {"Loss Field / (test/val)": np.mean(mse_losses), "L1 Loss Field / (test/val)": np.mean(l1_losses)}


def extract_mean_std(dataset, device='cpu'):
    def get_base_dataset(dataset):
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        return dataset

    base_dataset = get_base_dataset(dataset)
    out_mean, out_std = base_dataset.out_mean, base_dataset.out_std
    field_mean, field_std = base_dataset.field_mean, base_dataset.field_std
    return out_mean.to(device), out_std.to(device), field_mean.to(device), field_std.to(device)


def get_mean_from_field_solution(field_solution, field_mean, field_std, frequencies=None):
    v_ref = 1e-9
    eps = 1e-12
    B, n_frequencies = field_solution.shape[:2]
    # remove transformations in dataloader
    if frequencies is None:
        field_solution = field_solution * field_std + field_mean
    else:
        field_solution = field_solution * field_std + field_mean[frequencies].unsqueeze(-1).unsqueeze(-1)
    field_solution = torch.exp(field_solution) - eps
    # calculate frequency response
    v = torch.mean(field_solution.view(B, n_frequencies, -1), dim=-1)
    v = 10*torch.log10((v + 1e-12)/v_ref)
    return v.view(B, n_frequencies)
