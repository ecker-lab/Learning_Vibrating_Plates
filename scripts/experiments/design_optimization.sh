python scripts/run.py --model_cfg fqo_rn18.yaml --config cfg/V5000_smaller.yaml --dir transfer/bead_ratio/smaller/fqo_rn18 --batch_size 16
python scripts/run.py --model_cfg fqo_rn18.yaml --config cfg/V5000_larger.yaml --dir transfer/bead_ratio/larger/fqo_rn18 --batch_size 16


python scripts/run.py --model_cfg fqo_unet.yaml \
--config cfg/V5000.yaml --dir transfer/bead_ratio/smaller/fqo_unet  --batch_size 16 --filter_dataset smaller
python scripts/run.py --model_cfg fqo_unet.yaml \
--config cfg/V5000.yaml --dir transfer/bead_ratio/larger/fqo_unet   --batch_size 16 --filter_dataset larger
