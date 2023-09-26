import wandb, time, os, torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from acousticnn.plate.train import _evaluate
from acousticnn.plate.dataset import get_dataloader
from acousticnn.utils.builder import build_opti_sche
from acousticnn.utils.logger import init_train_logger, print_log
from acousticnn.utils.argparser import get_args, get_config
from torchinfo import summary
from acousticnn.plate.knn.knn_train import AutoEncoder, train_autoencoder, generate_encoding, get_checker, get_predictions, img_shape, pred_fn, get_output, eval_knn
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

k_max = 25

def main():
    args = get_args()
    config = get_config(args.config)
    logger = init_train_logger(args, config)
    config.dataset_keys = ["bead_patterns", "z_vel_mean_sq", "sample_mat"]

    trainloader, valloader, testloader, trainset, valset, testset = get_dataloader(args, config, logger, shuffle=False)
    net = AutoEncoder().cuda()
    image = next(iter(trainloader))["bead_patterns"]
    image = nn.functional.interpolate(image, img_shape) / 255
    print_log(summary(net, input_data=(image.to(args.device))), logger=logger)
    args.epochs = 300
    optimizer, scheduler = build_opti_sche(net, config)
    
    output = get_output(valset, config)
    print_log("pixel_space result", logger=logger)
    reference, queries = generate_encoding(trainloader, net, use_net=False), generate_encoding(valloader, net, use_net=False)
    net = train_autoencoder(args, net, trainloader, valloader, optimizer, scheduler, logger)




    output = get_output(valset, config)
    print_log("pixel_space result", logger=logger)
    reference, queries = generate_encoding(trainloader, net, use_net=False), generate_encoding(valloader, net, use_net=False)
    losses = eval_knn(reference, queries, k_max, config, query_set=valset, reference_set=trainset)
    k_opt = np.argmin(losses) + 1
    print_log("validation", logger=logger)
    print_log(k_opt, logger=logger)
    prediction = pred_fn(k_opt, trainset, trainloader, valloader, net, config, use_net=False)
    results = _evaluate(prediction, output, logger=None, config=None, args=None, epoch=None, report_peak_error=True, report_wasserstein=True, dataloader=valloader)
    r25, r75 = np.nanquantile(results["peak_ratio"], 0.25), np.nanquantile(results["peak_ratio"], 0.75)
    a,b,c = results["loss (test/val)"], results["wasserstein"], results["frequency_distance"]
    print_log(f"{a:4.2f} & {b:4.2f} & [{r25:3.2f}, {r75:3.2f}] & {c:3.1f}", logger=logger)


    print_log("\nautoencoder_result", logger=logger)
    reference, queries = generate_encoding(trainloader, net), generate_encoding(valloader, net)
    losses = eval_knn(reference, queries, k_max, config, query_set=valset, reference_set=trainset)
    k_opt = np.argmin(losses) + 1
    print_log("validation", logger=logger)
    print_log(k_opt, logger=logger)
    prediction = pred_fn(k_opt, trainset, trainloader, valloader, net, config, use_net=True)
    results = _evaluate(prediction, output, logger=None, config=None, args=None, epoch=None, report_peak_error=True, report_wasserstein=True, dataloader=valloader)
    r25, r75 = np.nanquantile(results["peak_ratio"], 0.25), np.nanquantile(results["peak_ratio"], 0.75)
    a,b,c = results["loss (test/val)"], results["wasserstein"], results["frequency_distance"]
    print_log(f"{a:4.2f} & {b:4.2f} & [{r25:3.2f}, {r75:3.2f}] & {c:3.1f}", logger=logger)



    print_log("\nautoencoder test", logger=logger)
    output = get_output(testset, config)
    print_log(k_opt, logger=logger)
    reference, queries = generate_encoding(trainloader, net), generate_encoding(testloader, net)
    prediction = pred_fn(k_opt, trainset, trainloader, testloader, net, config, use_net=True)
    results = _evaluate(prediction, output, logger=None, config=None, args=None, epoch=None, report_peak_error=True, report_wasserstein=True, dataloader=testloader)
    r25, r75 = np.nanquantile(results["peak_ratio"], 0.25), np.nanquantile(results["peak_ratio"], 0.75)
    a,b,c = results["loss (test/val)"], results["wasserstein"], results["frequency_distance"]
    print_log(f"{a:4.2f} & {b:4.2f} & [{r25:3.2f}, {r75:3.2f}] & {c:3.1f}", logger=logger)


if __name__ == '__main__':
    main()
