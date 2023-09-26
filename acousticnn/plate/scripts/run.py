import numpy as np
from acousticnn.plate.dataset import get_dataloader
from acousticnn.plate.model import model_factory
from acousticnn.utils.builder import build_opti_sche
from acousticnn.utils.logger import init_train_logger, print_log
from acousticnn.utils.argparser import get_args, get_config
from torchinfo import summary
import wandb, time, torch
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def main():
    args = get_args()

    config = get_config(args.config)
    model_cfg = get_config(args.model_cfg)
    config.dataset_keys = model_cfg.dataset_keys
    logger = init_train_logger(args, config)
    start_wandb(args, config)
    trainloader, valloader, testloader, trainset, valset, testset = get_dataloader(args, config, logger)
    net = model_factory(**model_cfg, conditional=config.conditional).to(args.device)

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
    results = evaluate(args, config, net, testloader, logger=logger, plot=False, report_peak_error=True, epoch=None, report_wasserstein=True)
    r25, r75 = np.nanquantile(results["peak_ratio"], 0.25), np.nanquantile(results["peak_ratio"], 0.75)
    a, b, c = results["loss (test/val)"], results["wasserstein"], results["frequency_distance"]
    print_log(f"{a:4.2f} & {b:4.2f} & [{r25:3.2f}, {r75:3.2f}] & {c:3.1f}", logger=logger)


def start_wandb(args, config):
    wandb.init(project="plate", entity="jans")
    wandb.config.update(config)
    wandb.run.name = args.dir_name + "_" + str(time.time())


if __name__ == '__main__':
    main()
