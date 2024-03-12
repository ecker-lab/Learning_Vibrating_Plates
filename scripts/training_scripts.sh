########################### ALL MODELS ###########################
python scripts/run.py --model_cfg localnet.yaml --config V5000.yaml --dir vibrating_plates/V5000/localnet --batch_size 16
python scripts/run.py --model_cfg localnet.yaml --config G5000.yaml --dir vibrating_plates/G5000/localnet --batch_size 16

python scripts/run.py --model_cfg unet.yaml --config V5000.yaml --dir vibrating_plates/V5000/unet --batch_size 16
python scripts/run.py --model_cfg unet.yaml --config G5000.yaml --dir vibrating_plates/G5000/unet --batch_size 16

python scripts/run.py --model_cfg vit_implicit.yaml --config V5000.yaml --dir vibrating_plates/V5000/vit_implicit
python scripts/run.py --model_cfg vit_implicit.yaml --config G5000.yaml --dir vibrating_plates/G5000/vit_implicit

python scripts/run.py --model_cfg query_rn18.yaml --config V5000.yaml --dir vibrating_plates/V5000/query_rn18 --batch_size 16
python scripts/run.py --model_cfg query_rn18.yaml --config G5000.yaml --dir vibrating_plates/G5000/query_rn18 --batch_size 16

python scripts/run.py --model_cfg grid_rn18.yaml --config V5000.yaml --dir vibrating_plates/V5000/grid_rn18 
python scripts/run.py --model_cfg grid_rn18.yaml --config G5000.yaml --dir vibrating_plates/G5000/grid_rn18

python scripts/run.py --model_cfg fno_decoder.yaml --config V5000.yaml --dir vibrating_plates/V5000/fno_decoder
python scripts/run.py --model_cfg fno_decoder.yaml --config G5000.yaml --dir vibrating_plates/G5000/fno_decoder

python scripts/run.py --model_cfg fno_fsm.yaml --config V5000.yaml --dir vibrating_plates/V5000/fno_fsm
python scripts/run.py --model_cfg fno_fsm.yaml --config G5000.yaml --dir vibrating_plates/G5000/fno_fsm

python knn/run_knn.py --config V5000.yaml --dir vibrating_plates/V5000/knn --batch_size 256
python knn/run_knn.py --config G5000.yaml --dir vibrating_plates/G5000/knn --batch_size 256

python scripts/run.py --model_cfg deeponet.yaml --config V5000.yaml --dir vibrating_plates/V5000/deeponet
python scripts/run.py --model_cfg deeponet.yaml --config G5000.yaml --dir vibrating_plates/G5000/deeponet


python scripts/run.py --model_cfg localnet.yaml --config V5000.yaml --dir vibrating_plates/data_variation/localnet/V5000/75_percent --wildcard 75 --batch_size 16 
python scripts/run.py --model_cfg localnet.yaml --config V5000.yaml --dir vibrating_plates/data_variation/localnet/V5000/50_percent --wildcard 50 --batch_size 16
python scripts/run.py --model_cfg localnet.yaml --config G5000.yaml --dir vibrating_plates/data_variation/localnet/G5000/75_percent --wildcard 75 --batch_size 16
python scripts/run.py --model_cfg localnet.yaml --config G5000.yaml --dir vibrating_plates/data_variation/localnet/G5000/50_percent --wildcard 50 --batch_size 16

python scripts/run.py --model_cfg localnet.yaml --config V5000.yaml --dir vibrating_plates/data_variation/localnet/V5000/25_percent --wildcard 25 --batch_size 16
python scripts/run.py --model_cfg localnet.yaml --config V5000.yaml --dir vibrating_plates/data_variation/localnet/V5000/10_percent --wildcard 10 --batch_size 16
python scripts/run.py --model_cfg localnet.yaml --config G5000.yaml --dir vibrating_plates/data_variation/localnet/G5000/25_percent --wildcard 25 --batch_size 16
python scripts/run.py --model_cfg localnet.yaml --config G5000.yaml --dir vibrating_plates/data_variation/localnet/G5000/10_percent --wildcard 10 --batch_size 16


python scripts/run.py --model_cfg query_rn18.yaml --config G5000.yaml --dir data_variation/query_rn18/G5000/75_percent --wildcard 75 --batch_size 16
python scripts/run.py --model_cfg query_rn18.yaml --config G5000.yaml --dir data_variation/query_rn18/G5000/50_percent --wildcard 50 --batch_size 16
python scripts/run.py --model_cfg query_rn18.yaml --config G5000.yaml --dir data_variation/query_rn18/G5000/25_percent --wildcard 25 --batch_size 16
python scripts/run.py --model_cfg query_rn18.yaml --config G5000.yaml --dir data_variation/query_rn18/G5000/10_percent --wildcard 10 --batch_size 16

python scripts/run.py --model_cfg query_rn18.yaml --config V5000.yaml --dir data_variation/query_rn18/V5000/75_percent --wildcard 75 --batch_size 16
python scripts/run.py --model_cfg query_rn18.yaml --config V5000.yaml --dir data_variation/query_rn18/V5000/50_percent --wildcard 50 --batch_size 16
python scripts/run.py --model_cfg query_rn18.yaml --config V5000.yaml --dir data_variation/query_rn18/V5000/25_percent --wildcard 25 --batch_size 16
python scripts/run.py --model_cfg query_rn18.yaml --config V5000.yaml --dir data_variation/query_rn18/V5000/10_percent --wildcard 10 --batch_size 16

python scripts/run.py --model_cfg query_rn18.yaml --config V5000_smaller.yaml --dir transfer/bead_ratio/smaller/query_rn18 --batch_size 16
python scripts/run.py --model_cfg query_rn18.yaml --config V5000_larger.yaml --dir transfer/bead_ratio/larger/query_rn18 --batch_size 16


python scripts/run.py --model_cfg localnet.yaml --config V5000_smaller.yaml --dir transfer/bead_ratio/smaller/localnet  --batch_size 16
python scripts/run.py --model_cfg localnet.yaml --config V5000_larger.yaml --dir transfer/bead_ratio/larger/localnet   --batch_size 16


