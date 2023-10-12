# Repository for: Vibroacoustic Frequency Response Prediction with Query-based Operator Networks
Preprint available from [arXiv](https://arxiv.org/abs/2310.05469).

![Methods image](readme_figure.png?raw=true "Title")

Understanding vibroacoustic wave propagation in mechanical structures like airplanes, cars and houses is crucial to ensure health and comfort of their users. To analyze such systems, designers and engineers primarily consider the dynamic response in the frequency domain, which is computed through expensive numerical simulations like the finite element method. In contrast, data-driven surrogate models offer the promise of speeding up these simulations, thereby facilitating tasks like design optimization, uncertainty quantification, and design space exploration. We present a structured benchmark for a representative vibroacoustic problem: Predicting the frequency response for vibrating plates with varying forms of beadings. The benchmark features a total of \nsamples plate geometries with an associated numerical solution and introduces evaluation metrics to quantify the prediction quality. 
To address the frequency response prediction task, we propose a novel frequency query operator model, which is trained to map plate geometries to frequency response functions. By integrating principles from operator learning and implicit models for shape encoding, our approach effectively addresses the prediction of resonance peaks of frequency responses. We evaluate the method on our vibrating-plates benchmark and find that it outperforms DeepONets, Fourier Neural Operators and more traditional neural network architectures.

## Data

The data is available from: https://doi.org/10.25625/UWF7RB

view_dataset.ipynb gives an example how to open the files saved in the hdf5 format and visualize the samples.

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
