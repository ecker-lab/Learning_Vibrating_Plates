import matplotlib.pyplot as plt
import numpy as np


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
    ax.plot(f, mean, lw=2.5)
    if quantile is not None:
        quantiles = np.quantile(losses_per_f, [0+quantile, 1-quantile], axis=0)
        ax.fill_between(f, quantiles[0], quantiles[1], alpha=0.2)

    ax.set_xlabel('Frequency')
    ax.set_ylabel('MSE')
    return ax