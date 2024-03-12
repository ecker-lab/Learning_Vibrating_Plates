import argparse
import munch
import yaml
import os
from acousticnn.plate.configs.main_dir import main_dir

def get_args(string_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.yaml", type=str, help='path to config file')
    parser.add_argument('--model_cfg', default="query_rn18.yaml", type=str, help='path to config file')
    parser.add_argument('--dir', default="debug", type=str, help='save directory')
    parser.add_argument('--device', default="cuda", type=str, help='choose cuda or cpu')
    parser.add_argument('--fp16', default="True", type=bool, help='use gradscaling')
    parser.add_argument('--seed', default="0", type=int, help='seed')
    parser.add_argument('--add_noise', action='store_true', help='add noise to beading pattern images during training')
    parser.add_argument('--alpha', default="0.9", type=float, help='alpha for loss weighting')

    parser.add_argument('--batch_size', default="64", type=int, help='batch_size')
    parser.add_argument('--wildcard', type=int, help='do anything with this argument')
    parser.add_argument('--ablation_cfg', default="None", type=str, help='specify ablation definition')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--continue_training', action='store_true', help='continue training from checkpoint')

    #'experiment args'
    if string_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(string_args)

    args.config = os.path.join(main_dir, "configs", args.config)
    args.model_cfg = os.path.join(main_dir, "configs/model_cfg/", args.model_cfg)
    args.ablation_cfg = os.path.join(main_dir, "configs/ablation_cfg/", args.ablation_cfg)
    args.dir_name = args.dir
    args.dir = os.path.join(main_dir, "experiments", args.dir)

    return args


def load_yaml_file(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)


def modify_default_cfg(default_config, main_cfg):
    for key, value in main_cfg.items():
        if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
            modify_default_cfg(default_config[key], value)
        else:
            default_config[key] = value

            
def get_config(config_path):
    # Get the main configuration file
    main_cfg = load_yaml_file(config_path)

    # Get the default arguments, which are paths to other YAML files
    default_args = main_cfg.pop('default_args', None)

    if default_args is not None:
        # Load the configuration from the default argument files
        for path in default_args:
            default_config = load_yaml_file(path)
            
            # Modify content of default args if specified so in main configuration and then update main configuration with modified default configuration
            for key in main_cfg:
                if key in default_config:
                    modify_default_cfg(default_config[key], main_cfg[key])
            
            main_cfg.update(default_config)
    return munch.Munch.fromDict(main_cfg)


def update_config(config, default_args):
    for key, val in default_args.items():
        sub_keys = key.split('.')
        sub_config = config
        for sub_key in sub_keys[:-1]:
            sub_config = sub_config[sub_key]
        sub_config[sub_keys[-1]] = val
    return config
