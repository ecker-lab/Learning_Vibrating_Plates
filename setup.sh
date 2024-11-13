conda create -n acoustics pytorch=2.5 torchvision=0.15 torchaudio=2.0 pytorch-cuda=11.7 timm=0.9 -c pytorch -c nvidia -c conda-forge 

conda activate acoustics
pip install munch==4.0 torchinfo==1.8 matplotlib==3.7 ipykernel==6.25 jupyter==1.0 transformers==4.35 scipy==1.14 wandb 

# to run fno and deeponet
pip install deepxde==1.10 neuraloperator==0.2 tensorly==0.8 tensorly-torch==0.4 opt_einsum==3.3 zarr==2.16

pip install h5py==3.10 hdf5plugin==4.2 seaborn==0.13 imageio==2.32
pip install git+https://github.com/aliutkus/torchinterp1d.git
pip install -e .

python -m ipykernel install --user --name=acoustics
