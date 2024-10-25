import numpy as np
import os
from acousticnn.plate.dataset import get_dataloader
from acousticnn.model import model_factory
from acousticnn.utils.logger import init_train_logger, print_log
from acousticnn.utils.argparser import get_args, get_config
from acousticnn.main_dir import wandb_project
from torchinfo import summary
import wandb, time, torch
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def main():
    args = get_args()

    config = get_config(args.config)
    model_cfg = get_config(args.model_cfg)
    logger = None
    trainloader, valloader, testloader, _, _, _ = get_dataloader(args, config, logger)
    net = model_factory(**model_cfg, conditional=config.conditional, rmfreqs=hasattr(config, "rmfreqs"), len_conditional=len(config.mean_conditional_param) if config.conditional else None).to(args.device)

    batch = next(iter(trainloader))
    batch = {k: v.to(args.device) for k, v in batch.items()}
    print_log(summary(net, input_data=(batch["bead_patterns"], batch["phy_para"], batch["frequencies"])), logger=logger)

    if model_cfg.velocity_field is True:
        from acousticnn.plate.train_fsm import train, evaluate
    else:
        from acousticnn.plate.train import train, evaluate

    data = torch.load(os.path.join(args.original_dir, "checkpoint_best"))
    new_state_dict = {}
    for key in data["model_state_dict"]:
        new_key = key.replace("_orig_mod.", "")  # Adjust the key as needed
        new_state_dict[new_key] = data["model_state_dict"][key]
    missing_keys, unexpected_keys = net.load_state_dict(new_state_dict, strict=False)
    print_log(f"missing keys: {missing_keys}, unexpected keys: {unexpected_keys}", logger=logger)
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

if __name__ == '__main__':
    main()
