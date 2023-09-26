import numpy as np
from acousticnn.plate.dataset import get_dataloader
from acousticnn.plate.model import model_factory
from acousticnn.utils.builder import build_opti_sche
from acousticnn.utils.logger import init_train_logger, print_log, close_logger
from acousticnn.utils.argparser import get_args, get_config, update_config
from torchinfo import summary
import os
import wandb, time, torch
from copy import deepcopy

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def main():
    args = get_args()
    config = get_config(args.config)
    model_cfg = get_config(args.model_cfg)
    base_dir = args.dir
    ablation_cfg = get_config(args.ablation_cfg)
    ablation_name = ablation_cfg.name
    ablation_args = ablation_cfg.args
    default_args = ablation_cfg.default_cfg_args if "default_cfg_args" in ablation_cfg else {}
    config = update_config(config, default_args)
    for key, vals in ablation_args.items():
        for val in vals:
            model_cfg_copy = deepcopy(model_cfg)
            sub_keys = key.split('.')
            sub_config = model_cfg_copy  # Notice this line
            for sub_key in sub_keys[:-1]:
                sub_config = sub_config[sub_key]
            sub_config[sub_keys[-1]] = val  # set new value
            print(model_cfg_copy)
            args.dir = os.path.join(base_dir, ablation_name, key, str(val))
            args.dir_name = os.path.join(ablation_name, key, str(val))
            one_ablation(args, config, model_cfg_copy)


def one_ablation(args, config, model_cfg):
    logger = init_train_logger(args, config)
    session = start_wandb(args, config)
    trainloader, valloader, testloader, trainset, valset, testset = get_dataloader(args, config, logger)
    print_log(model_cfg, logger=logger)
    net = model_factory(**model_cfg).to(args.device)

    if config.conditional is False:
        a = next(iter(trainloader))["bead_patterns"]
        print_log(summary(net, input_data=(a.to(args.device))), logger=logger)
    else:
        batch = next(iter(trainloader))
        a = batch["bead_patterns"].to(args.device)
        b = batch["sample_mat"].to(args.device)
        print_log(summary(net, input_data=(a, b)), logger=logger)
    optimizer, scheduler = build_opti_sche(net, config)

    if model_cfg.field_solution_map is True:
        from acousticnn.plate.train_fsm import train, evaluate
    else:
        from acousticnn.plate.train import train, evaluate

    net = train(args, config, net, trainloader, optimizer, valloader, scheduler=scheduler, logger=logger)
    results = evaluate(args, config, net, valloader, logger=logger, plot=False, report_peak_error=True, epoch=None, report_wasserstein=True)
    r25, r75 = np.quantile(results["peak_ratio"], 0.25), np.quantile(results["peak_ratio"], 0.75)
    a, b, c = results["loss (test/val)"], results["wasserstein"], results["frequency_distance"]
    print_log(f"{a:4.2f} & {b:4.2f} & [{r25:3.2f}, {r75:3.2f}] & {c:3.1f}", logger=logger)

    session.finish()
    close_logger(logger)


def start_wandb(args, config):
    wandb.init(project="plate", entity="jans")
    wandb.config.update(config)
    wandb.run.name = args.dir_name + "_" + str(time.time())
    return wandb.run


if __name__ == '__main__':
    main()
