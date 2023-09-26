import subprocess
import argparse
import queue
from threading import Thread

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='0', help='Comma-separated list of CUDA devices.')
    devices = parser.parse_args().cuda.split(',')
    print(devices)
    cmd_queue = queue.Queue()

    for model in ["query_unet"]: # "query_rn18", "grid_rn18", "fno_decoder", "vit_implicit",  "deeponet", "fno_fsm", "unet", "query_unet"
        for dataset in ["G5000.yaml", "fsm_V5000.yaml"]: # 
            cmd = f"python scripts/run.py --model_cfg {model}.yaml --config {dataset} --dir arch/{model}/{dataset.split('.')[0]}/original_lr"
            if model == "query_unet":
                cmd = cmd + " --batch_size 8"
            cmd_queue.put(cmd)
            print(cmd)
    threads = [Thread(target=run_on_device, args=(device, cmd_queue)) for device in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def run_on_device(device, cmd_queue):
    while not cmd_queue.empty():
        try:
            cmd = cmd_queue.get_nowait()
        except queue.Empty:
            return

        full_cmd = f"CUDA_VISIBLE_DEVICES={device} {cmd}"
        print(full_cmd)
        subprocess.run(full_cmd, shell=True)

if __name__ == "__main__":
    main()
