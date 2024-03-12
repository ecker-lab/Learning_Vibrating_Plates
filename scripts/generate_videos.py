import numpy as np
from acousticnn.plate.dataset import HDF5Dataset
from acousticnn.plate.model import model_factory
from acousticnn.utils.argparser import get_args, get_config
from acousticnn.utils.model_utils import get_field_prediction
from acousticnn.plate.configs.main_dir import main_dir
from acousticnn.plate.train_fsm import extract_mean_std
import wandb, time, torch, os, argparse
from acousticnn.utils.plot import plot_one_model, save_video, save_plot_at_peaks
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
normalize = False


def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    config = get_config(os.path.join(main_dir, "configs", args.config))
    model_cfg = get_config(os.path.join(main_dir, "configs/model_cfg/", args.model_cfg))
    dataset  = HDF5Dataset(args, config, config.data_paths_test, normalization=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, drop_last=False, shuffle=True)
    out_mean, out_std, field_mean, field_std = extract_mean_std(dataloader.dataset)

    net = model_factory(**model_cfg, conditional=config.conditional, n_frequencies=config.n_frequencies, len_conditional=len(config.mean_conditional_param) if config.conditional else None).cuda()
    net.load_state_dict(torch.load(args.ckpt)["model_state_dict"])

    for n_batch in range(args.n_batches):
        batch = next(iter(dataloader))
        actual_frequency_response, field_solution, image =  batch["z_vel_mean_sq"], batch["z_abs_velocity"], batch["bead_patterns"][:, 0]
        prediction, prediction_field = get_field_prediction(net, config, batch, dataloader)
        prediction_field = prediction_field
        if not normalize:
            prediction = prediction.mul(out_std).add_(out_mean)
            actual_frequency_response = actual_frequency_response.mul(out_std).add_(out_mean)
            prediction_field = dataset.undo_field_transformation(prediction_field.mul(field_std).add_(field_mean))
            field_solution = dataset.undo_field_transformation(field_solution.mul(field_std).add_(field_mean))
            "normalization removed"
        #print(actual_frequency_response.shape, prediction.shape)
        for i in range(len(batch)):
            plot_args = {"field_solution": field_solution, "frequency_response": actual_frequency_response, "field_prediction": prediction_field, "prediction": prediction,\
                            "geometries": image, "model": args.model_cfg, "scaling": args.scaling, "idx": i}
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
    
    
    args = parser.parse_args()
    main(args)

