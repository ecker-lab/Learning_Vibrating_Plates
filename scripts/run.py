import numpy as np
import os
from acousticnn.plate.dataset import get_dataloader
from acousticnn.model import model_factory
from acousticnn.utils.builder import build_opti_sche
from acousticnn.utils.logger import init_train_logger, print_log
from acousticnn.utils.argparser import get_args, get_config
from acousticnn.main_dir import wandb_project
from torchinfo import summary
import wandb, time, torch
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def main():
    args = get_args()
    print(args)
    config = get_config(args.config)
    model_cfg = get_config(args.model_cfg)
    if not hasattr(config, "lr"):
        config.optimizer.kwargs.lr = model_cfg.lr
    else:
        config.optimizer.kwargs.lr = config.lr
    if args.debug is True:
        model_cfg.dataset_keys = ["bead_patterns", "z_vel_mean_sq", "phy_para", 'frequencies']
    config.dataset_keys = model_cfg.dataset_keys
    if args.debug is False:
        logger = init_train_logger(args, config)
        start_wandb(args, config)
    else:
        logger = None

    trainloader, valloader, testloader, _, _, _ = get_dataloader(args, config, logger)
    net = model_factory(**model_cfg, conditional=config.conditional, len_conditional=len(config.mean_conditional_param) if config.conditional else None).to(args.device)
    if hasattr(config, 'use_both_datasets'):
        print_log("Using both datasets", logger=logger)
        args_new = get_args(["--config", f'cfg/V5000.yaml', "--model_cfg", args.model_cfg, '--batch_size', '4'])
        config_new = get_config(args_new.config)
        config_new.data_path_ref = config.data_path_ref
        config_new.dataset_keys = ["bead_patterns", "z_vel_mean_sq", 'z_vel_abs', 'frequencies']
        config_new.mean_conditional_param = [0.75, 0.5, 0.003, 0.02, 50, 0.5, 0.5]
        config_new.std_conditional_param = [0.086602546, 0.05773503, 0.0005773503, 0.005773503, 28.86751345948129, 0.17320508075688776,  0.17320508075688776]
        trainloader_v5000 = get_dataloader(args_new, config_new, logger)[0]
        trainloader = AlternatingDataLoader(trainloader, trainloader_v5000)

    batch = next(iter(trainloader))
    batch = {k: v.to(args.device) for k, v in batch.items()}
    print_log(summary(net, input_data=(batch["bead_patterns"], batch["phy_para"], batch["frequencies"])), logger=logger)
    optimizer, scheduler = build_opti_sche(net, config)

    if args.debug is True:
        return

    if model_cfg.velocity_field is True:
        from acousticnn.plate.train_fsm import train, evaluate
    else:
        from acousticnn.plate.train import train, evaluate


    if hasattr(config, "initial_checkpoint"):
        print_log(f"loading checkpoint from {config.initial_checkpoint}")
        data = torch.load(config.initial_checkpoint)
        new_state_dict = {}
        for key in data["model_state_dict"]:
            new_key = key.replace("_orig_mod.", "")  # Adjust the key as needed
            new_state_dict[new_key] = data["model_state_dict"][key]
        # load model and report key mismatches
        missing_keys, unexpected_keys = net.load_state_dict(new_state_dict, strict=False)
        print_log(f"missing keys: {missing_keys}, unexpected keys: {unexpected_keys}", logger=logger)

    if args.continue_training:
        data = torch.load(os.path.join(args.dir, "checkpoint_best"))
        new_state_dict = {}
        for key in data["model_state_dict"]:
            new_key = key.replace("_orig_mod.", "")  # Adjust the key as needed
            new_state_dict[new_key] = data["model_state_dict"][key]
        net.load_state_dict(new_state_dict)
        start_epoch = data["epoch"]
        optimizer.load_state_dict(data["optimizer_state_dict"])
        lowest_loss = 100
        scheduler.step(start_epoch)
        print_log(f"Continue training from epoch {start_epoch}, with loss {lowest_loss}", logger=logger)
        net = train(args, config, model_cfg, net, trainloader, optimizer, valloader, scheduler=scheduler, logger=logger, start_epoch=start_epoch, lowest_loss=lowest_loss)
    else:
        net = train(args, config, model_cfg, net, trainloader, optimizer, valloader, scheduler=scheduler, logger=logger)

    print_log(f"Evaluate on test set", logger=logger)
    results = evaluate(args, config, net, testloader, logger=logger, report_peak_error=True, epoch=None, report_wasserstein=True)
    a, b, c, save_rmean = results["loss (test/val)"], results["wasserstein"], results["frequency_distance"], results["save_rmean"]
    print_log(f"{a:4.2f} & {b:4.2f} & {save_rmean:3.2f} & {c:3.1f}", logger=logger)



    if args.filter_dataset != 'False':
        args.filter_dataset = reverse_orientation(args.filter_dataset)
        testloader2 = get_dataloader(args, config, logger)[2]
        print_log(f"Evaluate on reverse set {args.filter_dataset}", logger=logger)
        results = evaluate(args, config, net, testloader2, logger=logger, report_peak_error=True, epoch=None, report_wasserstein=True)
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



from itertools import cycle

class AlternatingDataLoader:
    def __init__(self, loader1, loader2):
        self.loader1 = loader1
        self.loader2 = loader2
        self.dataset = loader1.dataset
        self.switch = cycle([0, 1])

    def __iter__(self):
        self.iterators = (iter(self.loader1), iter(self.loader2))
        return self

    def __next__(self):
        iterator = self.iterators[self.switch.__next__()]
        try:
            batch = next(iterator)
            return batch
        except StopIteration:
            raise StopIteration  # If both loaders are exhausted


if __name__ == '__main__':
    main()
