import numpy as np
import os 
from acousticnn.plate.dataset import get_dataloader
from acousticnn.plate.model import model_factory
from acousticnn.utils.builder import build_opti_sche
from acousticnn.utils.logger import init_train_logger, print_log
from acousticnn.utils.argparser import get_args, get_config
from acousticnn.plate.configs.main_dir import wandb_project
from torchinfo import summary
import wandb, time, torch
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def main():
    args = get_args()

    config = get_config(args.config)
    model_cfg = get_config(args.model_cfg)
    config.optimizer.kwargs.lr = model_cfg.lr
    model_cfg.scaling_factor = args.wildcard
    args.wildcard = None
    if args.debug is True:
        model_cfg.dataset_keys = ["bead_patterns", "z_vel_mean_sq", "sample_mat"]
    config.dataset_keys = model_cfg.dataset_keys
    if args.debug is False:
        logger = init_train_logger(args, config)
        start_wandb(args, config)
    else:
        logger = None
    trainloader, valloader, testloader, trainset, valset, testset = get_dataloader(args, config, logger)
    net = model_factory(**model_cfg, conditional=config.conditional, n_frequencies=config.n_frequencies, rmfreqs=hasattr(config, "rmfreqs"), len_conditional=len(config.mean_conditional_param) if config.conditional else None).to(args.device)

    if config.conditional is False:
        a = next(iter(trainloader))["bead_patterns"]
        print_log(summary(net, input_data=(a.to(args.device))), logger=logger)
    else:
        batch = next(iter(trainloader))
        a = batch["bead_patterns"].to(args.device)
        b = batch["sample_mat"].to(args.device)
        print_log(summary(net, input_data=(a, b)), logger=logger)
    optimizer, scheduler = build_opti_sche(net, config)
    if args.debug is True:
        return
    if model_cfg.field_solution_map is True:
        from acousticnn.plate.train_fsm import train, evaluate
    else:
        from acousticnn.plate.train import train, evaluate
    if hasattr(config, "initial_checkpoint"):
        print_log(f"loading checkpoint from {config.initial_checkpoint}")
        data = torch.load(config.initial_checkpoint)
        net.load_state_dict(data["model_state_dict"])
    if args.continue_training:
        data = torch.load(os.path.join(args.dir, "checkpoint_last"))
        net.load_state_dict(data["model_state_dict"])
        start_epoch = data["epoch"]
        optimizer.load_state_dict(data["optimizer_state_dict"])
        #lowest_loss = data["loss"]
        lowest_loss = 100
        scheduler.step(start_epoch)
        print_log(f"Continue training from epoch {start_epoch}, with loss {lowest_loss}", logger=logger)
        net = train(args, config, net, trainloader, optimizer, valloader, scheduler=scheduler, logger=logger, start_epoch=start_epoch, lowest_loss=lowest_loss)
    else:
        net = train(args, config, net, trainloader, optimizer, valloader, scheduler=scheduler, logger=logger)
    print_log(f"Evaluate on test set", logger=logger)
    results = evaluate(args, config, net, testloader, logger=logger, plot=False, report_peak_error=True, epoch=None, report_wasserstein=True)
    a, b, c, save_rmean = results["loss (test/val)"], results["wasserstein"], results["frequency_distance"], results["save_rmean"]
    print_log(f"{a:4.2f} & {b:4.2f} & {save_rmean:3.2f} & {c:3.1f}", logger=logger)
    if config.filter_dataset is True:
        config.filter_orientation = reverse_orientation(config.filter_orientation)
        testloader2 = get_dataloader(args, config, logger)[2]
        print_log(f"Evaluate on reverse set {config.filter_orientation}", logger=logger)
        results = evaluate(args, config, net, testloader2, logger=logger, plot=False, report_peak_error=True, epoch=None, report_wasserstein=True)
        a, b, c, save_rmean = results["loss (test/val)"], results["wasserstein"], results["frequency_distance"], results["save_rmean"]
        print_log(f"{a:4.2f} & {b:4.2f} & {save_rmean:3.2f} & {c:3.1f}", logger=logger)


def reverse_orientation(filter_orientation):
    if filter_orientation == "smaller":
        return "larger"
    elif filter_orientation == "larger":
        return "smaller"
    else:
        raise ValueError(f'Unknown filter: {filter_orientation}')
        

def start_wandb(args, config):
    wandb.init(project=wandb_project)
    wandb.config.update(config)
    wandb.run.name = args.dir_name + "_" + str(time.time())


if __name__ == '__main__':
    main()
