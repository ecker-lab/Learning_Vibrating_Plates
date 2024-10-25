python scripts/run.py --model_cfg localnet.yaml --config limits/150k/50k3.yaml --dir limits/50k3 --batch_size 256
python scripts/run.py --model_cfg localnet.yaml --config limits/150k/15k10.yaml --dir limits/15k10 --batch_size 96
python scripts/run.py --model_cfg localnet.yaml --config limits/150k/10k15.yaml --dir limits/10k15 --batch_size 64

python scripts/run.py --model_cfg localnet.yaml --config limits/150k/5k30.yaml --dir limits/5k30 --batch_size 32
python scripts/run.py --model_cfg localnet.yaml --config limits/150k/2_5k60.yaml --dir limits/2_5k60 --batch_size 16
python scripts/run.py --model_cfg localnet.yaml --config limits/150k/1k150.yaml --dir limits/1k150 --batch_size 8
python scripts/run.py --model_cfg localnet.yaml --config limits/150k/05k300.yaml --dir limits/05k300 --batch_size 4
