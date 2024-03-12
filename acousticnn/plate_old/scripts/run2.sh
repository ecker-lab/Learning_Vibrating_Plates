# # frequency response architectures



# # Data Variation



CUDA_VISIBLE_DEVICES=5 python scripts/run.py --model_cfg query_unet.yaml --config G5000.yaml --dir data_variation/query_unet/G5000/75_percent --wildcard 75 --batch_size 8
CUDA_VISIBLE_DEVICES=5 python scripts/run.py --model_cfg query_unet.yaml --config G5000.yaml --dir data_variation/query_unet/G5000/50_percent --wildcard 50 --batch_size 8
CUDA_VISIBLE_DEVICES=5 python scripts/run.py --model_cfg query_unet.yaml --config G5000.yaml --dir data_variation/query_unet/G5000/25_percent --wildcard 25 --batch_size 8
CUDA_VISIBLE_DEVICES=5 python scripts/run.py --model_cfg query_unet.yaml --config G5000.yaml --dir data_variation/query_unet/G5000/10_percent --wildcard 10 --batch_size 8
# python scripts/run.py --model_cfg query_unet.yaml --config fsm_V5000.yaml --dir data_variation/query_unet/fsm_V5000/75_percent --wildcard 75
# python scripts/run.py --model_cfg query_unet.yaml --config fsm_V5000.yaml --dir data_variation/query_unet/fsm_V5000/50_percent --wildcard 50
# python scripts/run.py --model_cfg query_unet.yaml --config fsm_V5000.yaml --dir data_variation/query_unet/fsm_V5000/25_percent --wildcard 25
# python scripts/run.py --model_cfg query_unet.yaml --config fsm_V5000.yaml --dir data_variation/query_unet/fsm_V5000/10_percent --wildcard 10

# python run.py --model_cfg query_rn18.yaml --config F2500.yaml --dir data_variation/query_rn18/F2500/75_percent --wildcard 75
# python run.py --model_cfg query_rn18.yaml --config F2500.yaml --dir data_variation/query_rn18/F2500/50_percent --wildcard 50
# python run.py --model_cfg query_rn18.yaml --config F2500.yaml --dir data_variation/query_rn18/F2500/25_percent --wildcard 25
# python run.py --model_cfg query_rn18.yaml --config F2500.yaml --dir data_variation/query_rn18/F2500/10_percent --wildcard 10

# python run.py --model_cfg query_rn18.yaml --config V5000.yaml --dir data_variation/query_rn18/V5000/75_percent --wildcard 75
# python run.py --model_cfg query_rn18.yaml --config V5000.yaml --dir data_variation/query_rn18/V5000/50_percent --wildcard 50
# python run.py --model_cfg query_rn18.yaml --config V5000.yaml --dir data_variation/query_rn18/V5000/25_percent --wildcard 25
# python run.py --model_cfg query_rn18.yaml --config V5000.yaml --dir data_variation/query_rn18/V5000/10_percent --wildcard 10


# python run.py --model_cfg fno_decoder.yaml --config F2500.yaml --dir data_variation/fno_decoder/F2500/75_percent --wildcard 75
# python run.py --model_cfg fno_decoder.yaml --config F2500.yaml --dir data_variation/fno_decoder/F2500/50_percent --wildcard 50
# python run.py --model_cfg fno_decoder.yaml --config F2500.yaml --dir data_variation/fno_decoder/F2500/25_percent --wildcard 25
# python run.py --model_cfg fno_decoder.yaml --config F2500.yaml --dir data_variation/fno_decoder/F2500/10_percent --wildcard 10

# python run.py --model_cfg fno_decoder.yaml --config V5000.yaml --dir data_variation/fno_decoder/V5000/75_percent --wildcard 75
# python run.py --model_cfg fno_decoder.yaml --config V5000.yaml --dir data_variation/fno_decoder/V5000/50_percent --wildcard 50
# python run.py --model_cfg fno_decoder.yaml --config V5000.yaml --dir data_variation/fno_decoder/V5000/25_percent --wildcard 25
# python run.py --model_cfg fno_decoder.yaml --config V5000.yaml --dir data_variation/fno_decoder/V5000/10_percent --wildcard 10


# Field Solution Map Data

# python run.py --model_cfg fno.yaml --config fsm_V5000.yaml --dir arch/fno_fsm/fsm_V5000
# #python run.py --model_cfg unet.yaml --config fsm_V5000.yaml --dir arch/unet/fsm_V5000
# CUDA_VISIBLE_DEVICES=4 python run.py --model_cfg query_unet.yaml --config fsm_V5000.yaml --dir arch/query_unet/freqs5_k16_ --batch_size 8
# python run.py --model_cfg query_rn18.yaml --config fsm_V5000.yaml --dir arch/query_rn18/fsm_V5000

# #python run.py --model_cfg unet_conditional.yaml --config G5000.yaml --dir arch/unet/G5000
# python run.py --model_cfg fno_conditional.yaml --config G5000.yaml --dir arch/fno_fsm/G5000
# python run.py --model_cfg query_rn18_conditional.yaml --config G5000.yaml --dir arch/query_rn18/G5000

# # # Ablation studies

# #CUDA_VISIBLE_DEVICES=4 python run_ablation.py --model_cfg query_rn18.yaml --config fsm_V5000.yaml --dir ablation --ablation_cfg query_rn18_ablation.yaml

# #CUDA_VISIBLE_DEVICES=7 python run_ablation.py --model_cfg fno.yaml --config fsm_V5000.yaml --dir ablation --ablation_cfg fno_fsm_ablation.yaml

