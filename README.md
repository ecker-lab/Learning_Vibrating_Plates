# Repository for: Vibroacoustic Frequency Response Prediction with Query-based Operator Networks
Preprint available from [arXiv](https://arxiv.org/abs/2310.05469).

![Methods image](readme_figure.png?raw=true "Title")

Understanding vibroacoustic wave propagation in mechanical structures like airplanes, cars and houses is crucial to ensure health and comfort of their users. To analyze such systems, designers and engineers primarily consider the dynamic response in the frequency domain, which is computed through expensive numerical simulations like the finite element method. In contrast, data-driven surrogate models offer the promise of speeding up these simulations, thereby facilitating tasks like design optimization, uncertainty quantification, and design space exploration. We present a structured benchmark for a representative vibroacoustic problem: Predicting the frequency response for vibrating plates with varying forms of beadings. The benchmark features a total of \nsamples plate geometries with an associated numerical solution and introduces evaluation metrics to quantify the prediction quality. 
To address the frequency response prediction task, we propose a novel frequency query operator model, which is trained to map plate geometries to frequency response functions. By integrating principles from operator learning and implicit models for shape encoding, our approach effectively addresses the prediction of resonance peaks of frequency responses. We evaluate the method on our vibrating-plates benchmark and find that it outperforms DeepONets, Fourier Neural Operators and more traditional neural network architectures.


## Data

To download the data, we recommend using the script acousticnn/utils/download.py. Here we list out the commands to download the available dataset settings. Please note, that the root_folder must already exist:

| Setting        | Dataset Download                                             | Dataset Size |
| ----------- | :----------------------------------------------------------- | ------------ |
| small example file   | ```python acousticnn/utils/download.py --dataset_name single_example_G5000 --root_folder data/example``` | 2 GB        |
| V5000   | ```python acousticnn/utils/download.py --dataset_name reduced_resolution_V5000 --root_folder data/V-5000``` | 13 GB        |
| G5000   | ```python acousticnn/utils/download.py --dataset_name reduced_resolution_G5000 --root_folder data/G-5000``` | 13 GB        |
| V5000 (full resolution)   | ```python acousticnn/utils/download.py --dataset_name full_resolution_V5000 --root_folder data/full_resolution_V5000``` | 49 GB        |
| G5000 (full resolution)   | ```python acousticnn/utils/download.py --dataset_name full_resolution_G5000 --root_folder data/full_resolution_G5000``` | 49 GB        |


view_dataset.ipynb gives an example how to open the files saved in the hdf5 format and visualize the samples.


## Setup

1. To be able to reproduce the trainings and run the code, you need to set up a python environment with the required packages. We provide a setup script that set ups the environment based on conda. Run the following to set up the environment:

``
bash setup.sh
``

2. This repository employs [Weights and Biases](https://wandb.ai/) for logging. To be able to use it you must have an account and login by running:

``
wandb login
``

3. Further, you need to define the path, where you saved the data and the path of the repository. These are defined in the file  acousticnn/plate/configs/main_dir.py. Change data_dir to where you saved the data. Change main_dir to the root path of the repository, i.e. /user/xyz/repository. You can also specify the WandB project name you want to log to in this file.


## Train a model 

```
cd acousticnn/plate
python scripts/run.py --model_cfg query_rn18.yaml --config fsm_V5000.yaml --dir path/to/logs
```

Change the model_cfg and config args to specify the model architecture and dataset respectively. The config files are saved in acousticnn/plate/configs. --dir specifies the save and log directory within the folder acousticnn/plate/experiments.


## Evaluate a model 

Within acousticnn/plate use evaluate_fr.ipynb for frequency response prediction models or evaluate_fsm.ipynb for field solution map prediction models. 


## Acknowledgments

Parts of this code built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [PDEBench](https://github.com/pdebench/PDEBench) and [this repository](https://github.com/dome272/Diffusion-Models-pytorch). 
