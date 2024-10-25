import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import imageio
import seaborn as sns

def plot_results(prediction, amplitude, f, ax=None, quantile=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10 / 2.54, 8 / 2.54))

    ax.plot(f, amplitude,  label="Reference", color="#909090", lw=2.5,linestyle='dashed',dashes=[1, 1])
    ax.plot(f, prediction, alpha = 0.8,  label="Prediction", color="#e19c2c", lw=2.5)

    ax.set_ylim(-2.5, 5.5)
    ax.set_xlabel('frequency')
    ax.set_ylabel('normalized amplitude')
    ax.legend(fontsize=12)
    return ax


def plot_loss(losses_per_f, f, ax=None, quantile=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10 / 2.54, 8 / 2.54))
    mean = np.mean(losses_per_f, axis=0)
    ax.plot(f, mean, lw=0.5)
    if quantile is not None:
        quantiles = np.quantile(losses_per_f, [0+quantile, 1-quantile], axis=0)
        ax.fill_between(f, quantiles[0], quantiles[1], alpha=0.2)

    ax.set_xlabel('Frequency')
    ax.set_ylabel('MSE')
    return ax

### MAKE GIFS OR VIDEOS


def get_range(field_prediction, field_solution):
    a_min = np.amin(field_prediction, axis=(1, 2, 3))
    a_max = np.amax(field_solution, axis=(1, 2, 3))

    b_min = np.amin(field_prediction, axis=(1, 2, 3))
    b_max = np.amax(field_solution, axis=(1, 2, 3))
    min_values = np.minimum(a_min, b_min)
    max_values = np.maximum(a_max, b_max)
    return min_values, max_values


def plot_all_models(field_solution, frequency_response, field_prediction, prediction, geometries, models, model_names, frequency=0, idx=0, scaling=False):
    fig, axs = plt.subplots(len(models), 3, figsize=(15, int(4*len(models))))
    img = geometries[idx][::-1]
    y, x = np.mgrid[0:field_solution.shape[2], 0:field_solution.shape[3]]
    frequency_vector = np.arange(field_solution.shape[1])

    if scaling is True:
        min_values, max_values = get_range(field_prediction[models[0]], field_solution)
        vmin, vmax = min_values[idx], max_values[idx]
    else:
        vmin, vmax = None, None
    axs[0, 0].imshow(img, cmap=plt.cm.gray)
    axs[0, 0].set_title("Geometry")
    axs[0, 0].axis("off")
    axs[1, 0].contourf(x, y, field_solution[idx][frequency], vmin=vmin, vmax=vmax, levels=50, cmap=plt.cm.gray)
    axs[1, 0].set_title("Actual field solution")
    axs[1, 0].set_aspect("equal")
    axs[1, 0].axis("off")
    axs[2, 0].plot(frequency_vector, frequency_response[idx], color="#909090", linestyle='dashed')
    axs[2, 0].set_title("Actual frequency response")

    for i, (model, name) in enumerate(zip(models, model_names)):
        axs[i, 1].contourf(x, y, field_prediction[model][idx][frequency], vmin=vmin, vmax=vmax, levels=50, cmap=plt.cm.gray)
        axs[i, 1].set_title(f"{name} field solution prediction")
        axs[i, 1].set_aspect("equal")
        axs[i, 1].axis("off")
        axs[i, 2].plot(frequency_vector, prediction[model][idx], color="#55a78c", lw=1.5, alpha = 0.8, label='Prediction')
        axs[i, 2].set_title(f"{name} frequency response prediction")
        axs[i, 2].plot(frequency_vector, frequency_response[idx], color="#909090", linestyle='dashed', label="Reference")
        axs[i, 2].axvline(x=frequency, color='red', linestyle='--', label='Selected frequency')
        axs[i, 2].legend(fontsize=10, loc="upper left")
        axs[i, 2].grid(which="major")

    plt.tight_layout()
    fig.canvas.draw()
    return fig


def plot_one_model(field_solution, frequency_response, field_prediction, prediction, geometries, resolution, frequency, idx, scaling=False, **kwargs):
    fig, axs = plt.subplots(1, 4, figsize=np.array(resolution)[::-1]/100)
    img = geometries[idx]
    frequency_vector = np.arange(1, field_solution.shape[1] + 1)

    if scaling is True:
        min_values, max_values = get_range(field_prediction, field_solution)
        vmin, vmax = min_values[idx], max_values[idx]
        #print("do scaling")
    else:
        vmin, vmax = None, None
    axs[0].imshow(img, cmap=plt.cm.gray)
    axs[0].set_title("Geometry", fontsize=9)
    axs[0].axis("off")
    axs[1].imshow(field_solution[idx][frequency], vmin=vmin, vmax=vmax, cmap=plt.cm.gray)
    axs[1].set_title("Actual velocity field", fontsize=9)
    axs[1].set_aspect("equal")
    axs[1].axis("off")
    axs[2].imshow(field_prediction[idx][frequency], vmin=vmin, vmax=vmax,cmap=plt.cm.gray)
    axs[2].set_title(f"Predicted velocity field", fontsize=9)
    axs[2].set_aspect("equal")
    axs[2].axis("off")
    axs[3].plot(frequency_vector, prediction[idx], color="#55a78c", lw=1.5, alpha = 0.8, label='Prediction')
    axs[3].plot(frequency_vector, frequency_response[idx], color="#909090", linestyle='dashed', label="Reference")
    axs[3].set_title(f"Predicted frequency response", fontsize=9)
    axs[3].axvline(x=frequency, color='red', linestyle='--', label='Selected frequency')
    axs[3].legend(fontsize=7, loc="upper left")
    axs[3].set_ylim(-20, 80)
    axs[3].set_aspect(1.9)

    axs[3].set_yticks([-20, 0, 20, 40, 60, 80])
    axs[3].set_xticks([0, 100, 200, 300])

    sns.despine(trim=True, offset=7)
    axs[3].grid(which="major", lw=0.2)

    plt.tight_layout()
    return fig


def get_one_plot_as_img(plot_args):
    fig = plot_args["plot_fn"](**plot_args)
    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image_array, plot_args["frequency"]


def save_video(path, plot_args, max_freq, plot_fn, save_format=".mp4", resolution=None):
    plt.rc('font', size=8)
    if resolution is None:
        d = 4
        h, w = 1, 3
        resolution = int((h*d*0.81+ 0.10)*100),int(w*d*1.21*100)
    print("resolution", *resolution)
    ims = np.empty((max_freq, *resolution, 3),  dtype='uint8')
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(get_one_plot_as_img, {**plot_args, "frequency": frequency, "plot_fn": plot_fn, "resolution": resolution}) for frequency in range(max_freq)]
        for future in concurrent.futures.as_completed(futures):
            im, freq = future.result()
            ims[freq] = im
    print("imgs generated")
    if save_format == ".gif":
        imageio.mimsave(path+save_format, ims, fps=20, loop=0)
    elif save_format == ".mp4":
        writer = imageio.get_writer(path+save_format, fps=20)
        for img in ims:
            writer.append_data(np.array(img))
        writer.close()
    print("saved")


def plot_geometry(ax, plot_args):
    ax.imshow(plot_args["geometries"][plot_args["idx"]], cmap=plt.cm.gray)
    ax.set_title("Geometry")
    ax.axis("off")
    return ax

def plot_frequency_response(ax, plot_args):
    idx, actual_frequency_response, prediction = plot_args["idx"], plot_args["frequency_response"], plot_args["prediction"]
    ax.plot(actual_frequency_response[idx], lw=0.5, label="Reference", color="black", linestyle='dashed',)
    ax.plot(prediction[idx], lw=0.5, label="Prediction", color="#55a78c")
    ax.set_ylim(-20, 80)
    ax.grid(lw=0.2)
    ax.set_xlabel('Frequency', fontsize=5, labelpad=4)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_ylabel('Amplitude', fontsize=5, labelpad=3)
    sns.despine(ax=ax, offset=5)
    ax.legend()
    return ax


def save_plot_at_peaks(path, plot_args, max_freq):

    plt.rcParams['axes.labelsize'] = 5
    plt.rcParams['axes.titlesize'] = 5
    plt.rcParams['axes.titlesize'] = 5
    plt.rcParams.update({'font.size': 5})
    figsize = (6.75, 2.5)
    idx = plot_args["idx"]
    fig, axs = plt.subplots(2, 4, figsize=figsize)
    axs[0,0] = plot_geometry(axs[0,0], plot_args)
    axs[1,0] = plot_frequency_response(axs[1,0], plot_args)

    from acousticnn.plate.metrics import find_peaks
    actual_peaks, properties = find_peaks(plot_args["frequency_response"][idx])
    actual_peaks_predicted, properties = find_peaks(plot_args["prediction"][idx])
    actual_peaks = np.sort(actual_peaks[np.argsort(plot_args["frequency_response"][idx][actual_peaks])[-3:]])
    actual_peaks_predicted = np.sort(actual_peaks_predicted[np.argsort(plot_args["prediction"][idx][actual_peaks_predicted])[-3:]])

    field_prediction = plot_args["field_prediction"]
    field_solution = plot_args["field_solution"]
    if plot_args["scaling"] is True:
        min_values, max_values = get_range(field_prediction, field_solution)
        vmin, vmax = min_values[idx], max_values[idx]
    else:
        vmin, vmax = None, None

    def plot_fields(i, peak1, peak2, ax1, ax2):
        ax1.imshow(field_solution[idx][peak1], vmin=vmin, vmax=vmax, cmap=plt.cm.gray)
        ax1.axis("off")
        ax1.set_title(f"Actual velocity field for frequency {peak1}")
        ax2.imshow(field_prediction[idx][peak2], vmin=vmin, vmax=vmax, cmap=plt.cm.gray)
        ax2.set_title(f"Predicted velocity field for frequency {peak2}")
        ax2.axis("off")
        return ax1, ax2


    for i in range(np.min([len(actual_peaks), len(actual_peaks_predicted)])):
        plot_fields(i, actual_peaks[i], actual_peaks_predicted[i], axs[0, i+1], axs[1, i+1])

    axs[1, 1].set_position(axs[1, 1].get_position().translated(0, 0.04))
    axs[1, 2].set_position(axs[1, 2].get_position().translated(0, 0.04))
    axs[1, 3].set_position(axs[1, 3].get_position().translated(0, 0.04))
    plt.tight_layout()
    pos = axs[1, 0].get_position()
    new_width = pos.width * 0.7  # 40% smaller
    new_height = pos.height * 0.7  # 40% smaller
    new_left = pos.x0 + 0.04
    new_bottom = pos.y0 + 0.13
    axs[1, 0].set_position([new_left, new_bottom, new_width, new_height])
    plt.savefig(path + ".pdf", transparent=True)

    return fig