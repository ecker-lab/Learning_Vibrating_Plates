# Repository for: Learning to Predict Structural Vibrations

In mechanical structures like airplanes, cars and houses, noise is generated and transmitted through vibrations. To take measures to reduce this noise, vibrations need to be simulated with expensive numerical computations. Deep learning surrogate models present a promising alternative to classical numerical simulations as they can be evaluated magnitudes faster, while trading-off accuracy. To quantify such trade-offs systematically and foster the development of methods, we present a benchmark on the task of predicting the vibration of harmonically excited plates. The benchmark features a total of 12,000 plate geometries with varying forms of beadings, material, boundary conditions, load position and sizes with associated numerical solutions.
To address the benchmark task, we propose a new network architecture, named Frequency-Query Operator, which predicts vibration patterns of plate geometries given a specific excitation frequency. Applying principles from operator learning and implicit models for shape encoding, our approach effectively addresses the prediction of highly variable frequency response functions occurring in dynamic systems. To quantify the prediction quality, we introduce a set of evaluation metrics and evaluate the method on our vibrating-plates benchmark. Our method outperforms DeepONets, Fourier Neural Operators and more traditional neural network architectures and can be used for design optimization.

The paper is available from [openreview](https://openreview.net/forum?id=i4jZ6fCDdy&nesting=2&sort=date-desc).

# Update
* Our paper has been accepted to Neurips 2024! The updated paper is available from [openreview](https://openreview.net/forum?id=i4jZ6fCDdy&nesting=2&sort=date-desc).
* We recommend downloading the dataset from this [google drive folder](https://drive.google.com/drive/folders/1sJvcalpFBhYMamdMDpBx6PBU7QPjROL0?usp=drive_link).

## Data

We provide a notebook [here](notebooks/view_dataset.ipynb), that enables the quick and easy visualization of our dataset. The data is available from [google drive](https://drive.google.com/drive/folders/1sJvcalpFBhYMamdMDpBx6PBU7QPjROL0?usp=drive_link) in the hdf5 format. Alternately, the data is available from [this data repository](https://doi.org/10.25625/UWF7RB), but there were some reports with issues with downloading the files, therefore we recommend using the google drive link! There, we also provide information on the structure of the hdf5 files and how to access the data. 

https://github.com/user-attachments/assets/bf09cbf3-8e70-4919-9758-0c6f48a36e43


The video shows how the vibration patterns change for three example plates with the frequency. Changes in magnitude are not displayed. To download the data, from the dataset repository, the script acousticnn/utils/download.py can be used. However, we recommend using the [google drive](https://drive.google.com/drive/folders/1sJvcalpFBhYMamdMDpBx6PBU7QPjROL0?usp=drive_link) link, as there were some reports with issues when downloading the files. Here, we list out the commands to download the available dataset settings. Please note, that the root_folder must already exist:

| Setting        | Dataset Download                                             | Dataset Size |
| ----------- | :----------------------------------------------------------- | ------------ |
| small example file   | ```python acousticnn/utils/download.py --dataset_name single_example_G5000 --root_folder data/example``` | 4 GB        |
| V5000   | ```python acousticnn/utils/download.py --dataset_name V5000 --root_folder data/V5000``` | 21 GB        |
| G5000   | ```python acousticnn/utils/download.py --dataset_name G5000 --root_folder data/G5000``` | 21 GB        |
| Frequencies per plate experiment   | ```python acousticnn/utils/download.py --dataset_name V50k --root_folder data/V50k``` | 7 GB        |


## Setup

Given an installation of conda, run the following to setup the environment:
``
bash setup.sh
``

This repository employs [Weights and Biases](https://wandb.ai/) for logging. To be able to use it you must have an account and login:
``
wandb login
``

In acousticnn//main_dir.py change data_dir to where you saved the data. Change main_dir to the root path of the repository, i.e. /user/xyz/repository. In experiment dir, model checkpoints and logs for trainings are saved. You can specify the WandB project name you want to log to in this file as well.


## Train a model

``
python scripts/run.py --model_cfg fqo_unet.yaml --config cfg/V5000.yaml --dir path/to/logs --batch_size 16
``

Change the model_cfg and config args to specify the model and dataset respectively. --dir specifies the save and log directory within the experiment_dir. Please note that the available models are specified in configs/model_cfg. The calls for the experiments in our paper are specified in scripts/experiments.


## Evaluate a model

Use notebooks/evaluate.ipynb to generate plots and numerically evaluate already trained models.
To generate prediction videos:

``
python scripts/generate_videos.py --ckpt path/to/trained_model --model_cfg fqo_unet.yaml --config cfg/V5000.yaml --save_path plots/videos --scaling False --do_plots True
``

Change the model_cfg and config args to specify the model and dataset respectively and specify the path to a trained model checkpoint via --ckpt.
--save_path specifies where the resulting videos are saved.

## Design optimization

For generating novel plate designs with optimized vibrational properties please take a look at [this repository](https://github.com/ecker-lab/diffusion_minimizing_vibrations).

## Example results

The videos show example predictions for samples from the test set. For the first video for each dataset changes in magnitude are not displayed, which makes the changes more easily visible. For the second video for each dataset changes in magnitude are displayed, which means that many details are lost but the resonance frequencies can be seen.

### V5000 dataset

https://github.com/user-attachments/assets/e02738d3-b264-4a01-92c3-d6f24dda59fa

https://github.com/user-attachments/assets/4ba3480f-cfb1-4aaf-882d-8f02eb5a37ff

### G5000 dataset

https://github.com/user-attachments/assets/664ede54-94cb-40c9-bf4c-988116bf76d1

https://github.com/user-attachments/assets/cd74d98d-4f4c-4052-a8df-9c83a3329685


## Acknowledgments

Parts of this code are built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) and [PDEBench](https://github.com/pdebench/PDEBench).
