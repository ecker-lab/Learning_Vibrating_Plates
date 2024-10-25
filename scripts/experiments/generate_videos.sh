python scripts/generate_videos.py --ckpt /home/nimdelde/scratch/experiments/whisperplate/vibrating_plates/G5000/fqo_unet/20240731_152944/checkpoint_best \
--model_cfg fqo_unet.yaml --config cfg/G5000.yaml --save_path plots/videos/G5000/uniform --do_plots --n_batches 3
python scripts/generate_videos.py --ckpt /home/nimdelde/scratch/experiments/whisperplate/vibrating_plates/G5000/fqo_unet/20240731_152944/checkpoint_best \
--model_cfg fqo_unet.yaml --config cfg/G5000.yaml --save_path plots/videos/G5000/changing --do_plots --n_batches 3 --scaling True
python scripts/generate_videos.py --ckpt /home/nimdelde/scratch/experiments/whisperplate/vibrating_plates/V5000/fqo_unet/20240731_134309/checkpoint_best \
--model_cfg fqo_unet.yaml --config cfg/V5000.yaml --save_path plots/videos/V5000/uniform --do_plots --n_batches 3
python scripts/generate_videos.py --ckpt /home/nimdelde/scratch/experiments/whisperplate/vibrating_plates/V5000/fqo_unet/20240731_134309/checkpoint_best \
--model_cfg fqo_unet.yaml --config cfg/V5000.yaml --save_path plots/videos/V5000/changing --do_plots --n_batches 3 --scaling True
python scripts/generate_videos.py --ckpt /home/nimdelde/scratch/experiments/whisperplate/legacy/V50k/50k15/20240806_151153/checkpoint_best \
--model_cfg fqo_unet.yaml --config cfg/V5000.yaml --save_path plots/videos/V50k/uniform --do_plots --n_batches 3
python scripts/generate_videos.py --ckpt /home/nimdelde/scratch/experiments/whisperplate/legacy/V50k/50k15/20240806_151153/checkpoint_best \
--model_cfg fqo_unet.yaml --config cfg/V5000.yaml --save_path plots/videos/V50k/changing --do_plots --n_batches 3 --scaling True
python scripts/generate_videos.py --ckpt /home/nimdelde/scratch/experiments/whisperplate/transfer/G5000_V5000_parallel/20241022_170009/checkpoint_best \
--model_cfg fqo_unet.yaml --config cfg/G5000.yaml --save_path plots/videos/G5000_transfer/uniform --do_plots --n_batches 3
python scripts/generate_videos.py --ckpt /home/nimdelde/scratch/experiments/whisperplate/transfer/G5000_V5000_parallel/20241022_170009/checkpoint_best \
--model_cfg fqo_unet.yaml --config cfg/G5000.yaml --save_path plots/videos/G5000_transfer/changing --do_plots --n_batches 3 --scaling True
