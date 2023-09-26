conda env remove -n acoustics
conda create -n acoustics pytorch torchvision torchaudio pytorch-cuda=11.7 timm -c pytorch -c nvidia -c conda-forge 

conda activate acoustics
pip install munch torchinfo matplotlib ipykernel jupyter transformers scipy wandb 

# to run fno and deeponet
pip insall deepxdeneuraloperator tensorly tensorly-torch opt_einsum zarr

pip install h5py hdf5plugin
pip install -e .

python -m ipykernel install --user --name=acoustics

