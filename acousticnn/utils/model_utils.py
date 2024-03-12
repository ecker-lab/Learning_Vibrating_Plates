from acousticnn.plate.model import model_factory
import os, torch
import numpy as np
from acousticnn.plate.configs.main_dir import main_dir
from acousticnn.plate.train_fsm import extract_mean_std, get_mean_from_field_solution
from acousticnn.utils.argparser import get_args, get_config
from acousticnn.plate.train import _generate_preds
from acousticnn.plate.train import evaluate, _generate_preds
from acousticnn.plate.train_fsm import evaluate as evaluate_fsm

base_path = os.path.join(main_dir, "experiments")


def get_net(model, conditional, len_conditional=4):
    model_cfg = model + ".yaml"
    args = get_args(["--config", "0toy.yaml", "--model_cfg", model_cfg])
    return model_factory(**get_config(os.path.join(main_dir, args.model_cfg)), conditional=conditional, len_conditional=len_conditional)


def get_field_prediction(model, config, batch, dataloader, path=None, max_freq=300, device="cuda"):
    if isinstance(model, str):
        net = get_net(model, conditional=config.conditional, len_conditional=len(config.mean_conditional_param) if config.conditional else None).to(device)
        net.load_state_dict(torch.load(path)["model_state_dict"])
        net.eval()
    else: 
        net = model
    with torch.no_grad():
        out_mean, out_std, field_mean, field_std = extract_mean_std(dataloader.dataset)
        out_mean, out_std = out_mean.to(device), out_std.to(device)
        field_mean, field_std = field_mean.to(device), field_std.to(device)
        image, field_solution, output, condition = batch["bead_patterns"], batch["z_abs_velocity"], batch["z_vel_mean_sq"], batch["sample_mat"]
        image, field_solution, output, condition = image.to(device), field_solution.to(device), output.to(device), condition.to(device)
        prediction_field = net(image, condition)
        pred_field = prediction_field.clone()
        prediction = get_mean_from_field_solution(prediction_field, field_mean, field_std)
        prediction.sub_(out_mean).div_(out_std)
    return prediction.cpu()[:, :max_freq], pred_field.cpu()[:, :max_freq]


def get_freq_prediction(model, config, batch, path=None, max_freq=300, device="cuda"):
    if isinstance(model, str):
        net = get_net(model, conditional=config.conditional, len_conditional=len(config.mean_conditional_param) if config.conditional else None).cuda()
        net.load_state_dict(torch.load(path)["model_state_dict"])
        net.eval()
    else:
        net = model
    with torch.no_grad():
        image, output, condition = batch["bead_patterns"], batch["z_vel_mean_sq"], batch["sample_mat"]
        image, output, condition = image.to(device), output.to(device), condition.to(device)
        prediction = net(image, condition)
    return prediction.cpu()[:, :max_freq]

