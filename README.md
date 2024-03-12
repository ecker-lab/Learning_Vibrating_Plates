# Repository for: Learning to Predict Structural Vibrations
Preprint available from [arXiv](https://arxiv.org/abs/2310.05469).

## Update

We are in the process of updating this repository. 


## Data

The data is available from: https://doi.org/10.25625/UWF7RB

The video shows how the vibration patterns change for three example plates with the frequency. Changes in magnitude are not displayed. To download the data, we recommend using the script acousticnn/utils/download.py. Here we list out the commands to download the available dataset settings. Please note, that the root_folder must already exist:


| Setting        | Dataset Download                                             | Dataset Size |
| ----------- | :----------------------------------------------------------- | ------------ |
| small example file   | ```python acousticnn/utils/download.py --dataset_name single_example_G5000 --root_folder data/example``` | 2 GB        |
| V5000   | ```python acousticnn/utils/download.py --dataset_name V5000 --root_folder data/V5000``` | 13 GB        |
| G5000   | ```python acousticnn/utils/download.py --dataset_name G5000 --root_folder data/G5000``` | 13 GB        |


notebooks/view_dataset.ipynb gives an example how to open the files saved in the hdf5 format and visualize the samples.


## Setup

Given an installation of conda, run the following to setup the environment:
``
bash setup.sh
``

This repository employs [Weights and Biases](https://wandb.ai/) for logging. To be able to use it you must have an account and login:
``
wandb login
``

In acousticnn/plate/configs/main_dir.py change data_dir to where you saved the data. Change main_dir to the root path of the repository, i.e. /user/xyz/repository. You can specify the WandB project name you want to log to in this file as well.


## Train a model 

``
python scripts/run.py --model_cfg query_rn18.yaml --config V5000.yaml --dir path/to/logs
``

Change the model_cfg and config args to specify the model and dataset respectively. --dir specifies the save and log directory within the folder acousticnn/plate/experiments. Please note that the available models are specified in acousticnn/plate/configs/model_cfg. The best model in our experiments, FQO-UNet, is specified as localnet.yaml.


## Evaluate a model 

Use notebooks/evaluate.ipynb to generate plots and numerically evaluate already trained models.
To generate prediction videos: 

``
python scripts/generate_videos.py --ckpt path/to/trained_model --model_cfg localnet.yaml --config V5000.yaml --save_path plots/videos

``
Change the model_cfg and config args to specify the model and dataset respectively and specify the path to a trained model checkpoint via --ckpt. 
--save_path specifies where the resulting videos are saved.


## Acknowledgments

Parts of this code are built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) and [PDEBench](https://github.com/pdebench/PDEBench).