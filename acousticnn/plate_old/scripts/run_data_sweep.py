import subprocess

device="CUDA_VISIBLE_DEVICES=4"

data_configs = ["fsm_V5000.yaml", "G5000.yaml"]
models = ["vit_implicit", "query_rn18", "grid_rn18", "deeponet", "fno_decoder"]

for model in models:
    for dataset in data_configs:
        cmd = f"{device} python run.py --model_cfg {model}.yaml --config {dataset} --dir arch/{model}/{dataset.split('.')[0]}"
        print(cmd)
        #subprocess.run(cmd, shell=True)

# Run KNN separately
# for config in data_configs:
#     cmd = ["python", "knn/run_knn.py", "--config", config, "--dir", f"arch/knn/{config.split('.')[0]}"]
#     subprocess.run(cmd)