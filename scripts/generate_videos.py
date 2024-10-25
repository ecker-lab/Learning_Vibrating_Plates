import numpy as np
from acousticnn.plate.dataset import HDF5Dataset
from acousticnn.model import model_factory
from acousticnn.utils.argparser import get_config
from acousticnn.main_dir import main_dir
from acousticnn.plate.train_fsm import extract_mean_std, get_mean_from_field_solution
import torch, os, argparse
from acousticnn.utils.plot import plot_one_model, save_video, save_plot_at_peaks
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
normalize = False
fields = ["bead_patterns", "z_vel_abs", "z_vel_mean_sq", "phy_para", "frequencies"]
batch_size = 2

def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    config = get_config(os.path.join(main_dir, "configs", args.config))
    model_cfg = get_config(os.path.join(main_dir, "configs/model_cfg/", args.model_cfg))
    dataset  = HDF5Dataset(args, config, config.data_paths_test, test=True, normalization=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    out_mean, out_std, field_mean, field_std = extract_mean_std(dataloader.dataset, device=args.device)

    net = model_factory(**model_cfg, conditional=config.conditional, len_conditional=len(config.mean_conditional_param) if config.conditional else None).cuda()
    data = torch.load(os.path.join(args.ckpt))
    new_state_dict = {}
    for key in data["model_state_dict"]:
        new_key = key.replace("_orig_mod.", "")  # Adjust the key as needed
        new_state_dict[new_key] = data["model_state_dict"][key]
    missing_keys, unexpected_keys = net.load_state_dict(new_state_dict, strict=False)

    for n_batch in range(args.n_batches):
        batch = next(iter(dataloader))
        image, field_solution, vel_mean_sq, condition, frequencies = (batch[field].to(args.device) for field in fields)
        with torch.no_grad():
            prediction_field = net(image, condition, frequencies)
        prediction = get_mean_from_field_solution(prediction_field, field_mean, field_std, frequencies)
        prediction.sub_(out_mean[frequencies]).div_(out_std)
        if not normalize:
            prediction = prediction.mul(out_std).add_(out_mean[frequencies])
            vel_mean_sq = vel_mean_sq.mul(out_std).add_(out_mean[frequencies])
            prediction_field = dataset.undo_field_transformation(prediction_field.mul(field_std).add_(field_mean[frequencies].unsqueeze(-1).unsqueeze(-1)))
            field_solution = dataset.undo_field_transformation(field_solution.mul(field_std).add_(field_mean[frequencies].unsqueeze(-1).unsqueeze(-1)))
            "normalization removed"
        prediction, prediction_field, field_solution, vel_mean_sq, image = prediction.cpu().numpy(), prediction_field.cpu().numpy(), \
                                                                field_solution.cpu().numpy(), vel_mean_sq.cpu().numpy(), image.cpu().numpy()
        #print(actual_frequency_response.shape, prediction.shape)
        for i in range(batch_size):
            plot_args = {"field_solution": field_solution, "frequency_response": vel_mean_sq, "field_prediction": prediction_field, "prediction": prediction,\
                            "geometries": image[:,0], "model": args.model_cfg, "scaling": args.scaling, "idx": i}
            #print(plot_args)
            save_video(os.path.join(args.save_path, f"video_{n_batch}.{i}"), plot_args, 300, plot_one_model, save_format=".mp4", resolution=(336, 1456))

            if args.do_plots:
                save_plot_at_peaks(os.path.join(args.save_path, f"plot_{n_batch}.{i}"), plot_args, 300)

if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser(description='Generate videos')
    parser.add_argument('--save_path', default="plots/videos", type=str, help='path to save videos')
    parser.add_argument('--ckpt', type=str, help='model checkpoint for predictions')
    parser.add_argument('--config', default="V5000.yaml", type=str, help='data_config')
    parser.add_argument('--model_cfg', default="localnet.yaml", type=str, help='model_config')
    parser.add_argument('--n_batches', default=1, type=int, help='number of batches of videos to generate')
    parser.add_argument('--scaling', default=False, type=bool, help='scale plots')
    parser.add_argument('--max_freq', type=int, help='max frequency to plot')
    parser.add_argument('--do_plots', action='store_true')
    parser.add_argument('--device', type=str, default="cuda")


    args = parser.parse_args()
    main(args)
