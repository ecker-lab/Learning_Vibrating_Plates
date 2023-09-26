# Repository for: Vibroacoustic Frequency Response Prediction with Query-based Operator Networks

This repository is in a preliminary stage. Additional documentation will be added soon.

## Data and pretrained models

Data available from: to.be.published

## Setup

in acousticnn/plate/configs/main_dir.py change data_dir to where you saved the data. Change main_dir to the root path of the repository, i.e. /user/xyz/repository.

Given an installation of conda, run the following to setup the environment:
``
bash setup.sh
``

## Train a model 

``
cd acousticnn/plate
python run.py --model_cfg query_rn18.yaml --config fsm_V5000.yaml --dir path/to/logs
``

Change the model_cfg and config args to specify the model and dataset respectively. --dir specifies the save and log directory within the folder acousticnn/plate/experiments.


## Evaluate a model 

Within acousticnn/plate use evaluate_fr.ipynb for frequency response prediction models or evaluate_fsm.ipynb for field solution map prediction models. 
