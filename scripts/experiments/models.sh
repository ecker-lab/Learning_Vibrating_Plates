python scripts/run.py --config cfg/V5000.yaml --model_cfg fqo_unet.yaml --dir vibrating_plates/V5000/fqo_unet --batch_size 16
python scripts/run.py --config cfg/V5000_no_sampling.yaml --model_cfg fqo_rn18.yaml --dir vibrating_plates/V5000/fqo_rn18 --batch_size 16
python scripts/run.py --config cfg/V5000_no_sampling.yaml --model_cfg fqo_vit.yaml --dir vibrating_plates/V5000/fqo_vit --batch_size 16
python scripts/run.py --config cfg/V5000_no_sampling.yaml --model_cfg grid_rn18.yaml --dir vibrating_plates/V5000/grid_rn18
python scripts/run.py --config cfg/V5000_no_sampling.yaml --model_cfg grid_unet.yaml --dir vibrating_plates/V5000/grid_unet --batch_size 16
python scripts/run.py --config cfg/V5000_no_sampling.yaml --model_cfg fno_fsm.yaml --dir vibrating_plates/V5000/fno_fsm --fp16 False --compile False
python  scripts/run.py --config cfg/V5000_no_sampling.yaml --model_cfg fno_decoder.yaml --dir vibrating_plates/V5000/fno_decoder --fp16 False
python scripts/run.py --config cfg/V5000_no_sampling.yaml --model_cfg deeponet.yaml --dir vibrating_plates/V5000/deeponet

python scripts/run.py --config cfg/G5000.yaml --model_cfg fqo_unet.yaml --dir vibrating_plates/G5000/fqo_unet --batch_size 16
python scripts/run.py --config cfg/G5000_no_sampling.yaml --model_cfg fqo_vit.yaml --dir vibrating_plates/G5000/fqo_vit --batch_size 16
python scripts/run.py --config cfg/G5000_no_sampling.yaml --model_cfg fqo_rn18.yaml --dir vibrating_plates/G5000/fqo_rn18 --batch_size 16
python scripts/run.py --config cfg/G5000_no_sampling.yaml --model_cfg grid_rn18.yaml --dir vibrating_plates/G5000/grid_rn18
python scripts/run.py --config cfg/G5000_no_sampling.yaml --model_cfg grid_unet.yaml --dir vibrating_plates/G5000/grid_unet --batch_size 16
python scripts/run.py --config cfg/G5000_no_sampling.yaml --model_cfg fno_fsm.yaml --dir vibrating_plates/G5000/fno_fsm --fp16 False --compile False
python scripts/run.py --config cfg/G5000_no_sampling.yaml --model_cfg fno_decoder.yaml --dir vibrating_plates/G5000/fno_decoder --fp16 False
python scripts/run.py --config cfg/G5000_no_sampling.yaml --model_cfg deeponet.yaml --dir vibrating_plates/G5000/deeponet --debug

python scripts/run_knn.py --config cfg/V5000_no_sampling.yaml --dir vibrating_plates/V5000/knn --batch_size 256
python scripts/run_knn.py --config cfg/G5000_no_sampling.yaml --dir vibrating_plates/G5000/knn --batch_size 256
