from acousticnn.model import model_factory
import os, torch
import numpy as np
from acousticnn.main_dir import main_dir
from acousticnn.utils.argparser import get_args, get_config
base_path = os.path.join(main_dir, "experiments")


def get_net(model, conditional, len_conditional=4):
    model_cfg = model + ".yaml"
    args = get_args(["--config", "0toy.yaml", "--model_cfg", model_cfg])
    return model_factory(**get_config(os.path.join(main_dir, args.model_cfg)), conditional=conditional, len_conditional=len_conditional)
