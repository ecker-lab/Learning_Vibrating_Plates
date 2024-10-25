import argparse
import munch
import yaml
import os
from acousticnn.main_dir import main_dir, experiment_dir
import time

def get_args(string_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.yaml", type=str, help='path to config file')
    parser.add_argument('--model_cfg', default="query_rn18.yaml", type=str, help='path to config file')
    parser.add_argument('--dir', default="debug", type=str, help='save directory')
    parser.add_argument('--device', default="cuda", type=str, help='choose cuda or cpu')
    parser.add_argument('--fp16', choices=[True, False], type=lambda x: x == 'True', default=True, help='use gradscaling (True/False)')
    parser.add_argument('--compile', choices=[True, False], type=lambda x: x == 'True', default=True, help='compile network (True/False)')
    parser.add_argument('--filter_dataset', choices=['smaller', 'larger', 'False'], type=str, default='False', help='filter dataset for transfer experiment')
    parser.add_argument('--seed', default="0", type=int, help='seed')
    parser.add_argument('--add_noise', action='store_true', help='add noise to beading pattern images during training')
    parser.add_argument('--alpha', default="0.9", type=float, help='alpha for loss weighting')

    parser.add_argument('--batch_size', default="64", type=int, help='batch_size')
    parser.add_argument('--wildcard', type=int, help='do anything with this argument')
    parser.add_argument('--ablation_cfg', default="None", type=str, help='specify ablation definition')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--load_idx', action='store_true', help='debug mode')
    parser.add_argument('--continue_training', action='store_true', help='continue training from checkpoint')

    #'experiment args'
    if string_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(string_args)

    args.config = os.path.join(main_dir, "configs", args.config)
    args.model_cfg = os.path.join(main_dir, "configs/model_cfg/", args.model_cfg)
    args.dir_name = args.dir
    args.original_dir = os.path.join(experiment_dir, args.dir)
    args.dir = os.path.join(experiment_dir, args.dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))

    return args


def load_yaml_file(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)


def get_config(config_path):
    # Get the main configuration file
    main_cfg = load_yaml_file(config_path)

    # Get the default arguments, which are paths to other YAML files
    default_args = main_cfg.pop('default_args', None)

    if default_args is not None:
        # Load the configuration from the default argument files
        for path in default_args:
            default_config = load_yaml_file(path)

            # Modify content of default config if different entry in main config and then update main config with modified default config
            for key in main_cfg:
                if key in default_config:
                    default_config[key] = main_cfg[key]

            main_cfg.update(default_config)
    return munch.Munch.fromDict(main_cfg)
