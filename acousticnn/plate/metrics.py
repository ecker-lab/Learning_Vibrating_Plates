from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
import numpy as np


def detect_peaks(spectrum, prominence=0.5):
    actual_peaks, properties = find_peaks(spectrum, prominence=prominence)
    return actual_peaks, properties


def backtransform(spectrum, mean, std, do_power=True):
    mean = mean.array.cpu().numpy()[:spectrum.shape[1]]
    spectrum = (spectrum * std.cpu().numpy() + mean)
    if do_power:
        spectrum = np.power(10, spectrum * 1e-1)
    return spectrum


def compute_wasserstein_distance(spectrum1, spectrum2, mean, std):
    """Compute the Wasserstein distance between two power spectra."""
    n_samples, n_freqs = spectrum1.shape
    distances = []
    spectrum1 = backtransform(spectrum1.copy(), mean, std)
    spectrum2 = backtransform(spectrum2.copy(), mean, std)
    if spectrum1.shape != spectrum2.shape:
        raise ValueError("Spectra must have the same shape.")
    if (spectrum1 < 0).any() or (spectrum2 < 0).any():
        raise ValueError("Spectra must contain non-negative values.")
    if spectrum1.sum() == 0 or spectrum2.sum() == 0:
        raise ValueError("Spectra sums must be greater than zero.")
    freqs = np.arange(0, n_freqs)
    # Normalize the spectra to make them into probability distributions
    spectrum1 /= spectrum1.sum()
    spectrum2 /= spectrum2.sum()

    for i in range(n_samples):
        distances.append(wasserstein_distance(freqs, freqs, spectrum1[i], spectrum2[i]))
    return np.mean(distances)


def compute_peak_distances(actual_amplitudes, predicted_amplitudes, actual_frequencies, predicted_frequencies):
    # Compute amplitude and frequency differences using broadcasting
    amplitude_diffs = np.abs(actual_amplitudes[:, None] - predicted_amplitudes)
    frequency_diffs = np.abs(actual_frequencies[:, None] - predicted_frequencies)

    # Compute distance matrix using given weights
    distance_matrix = frequency_diffs

    return distance_matrix, amplitude_diffs, frequency_diffs


def _peak_frequency_error(actual_amplitudes, predicted_amplitudes, prominence_threshold=0.5):
    # Find peaks
    actual_peaks, _ = find_peaks(actual_amplitudes, prominence=prominence_threshold, wlen=100)
    predicted_peaks, _ = find_peaks(predicted_amplitudes, prominence=prominence_threshold, wlen=100)

    # Get peak amplitudes
    actual_peak_amplitudes = actual_amplitudes[actual_peaks]
    predicted_peak_amplitudes = predicted_amplitudes[predicted_peaks]
    # Compute distance matrix
    distance_matrix, amplitude_diffs, frequency_diffs = compute_peak_distances(actual_peak_amplitudes, predicted_peak_amplitudes, actual_peaks,
                                                                               predicted_peaks)

    # Perform optimal assignment
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # Compute mean distance for matched peaks
    matched_amplitude_distance = np.mean(amplitude_diffs[row_indices, col_indices])
    matched_frequency_distance = np.mean(frequency_diffs[row_indices, col_indices])
    # Compute number of non-matched peaks
    peak_ratio = np.abs(len(predicted_peaks)) / len(actual_peaks)
    save_peak_ratio = np.min((peak_ratio, len(actual_peaks) / np.abs(len(predicted_peaks))))
    return peak_ratio, matched_amplitude_distance, matched_frequency_distance, len(actual_peaks), save_peak_ratio


def peak_frequency_error(actual_amplitudes, predicted_amplitudes, prominence_threshold=0.5):
    # Number of samples
    n_samples = actual_amplitudes.shape[0]
    ratios, save_ratios, n_peaks = [], [], []
    amplitude_distance, frequency_distance = [], []

    # Loop over samples
    for i in range(n_samples):
        peak_ratio, matched_amplitude_distance, matched_frequency_distance, n_peak, save_peak_ratio = _peak_frequency_error(
            actual_amplitudes[i], predicted_amplitudes[i], prominence_threshold
        )
        ratios.append(peak_ratio), n_peaks.append(n_peak), save_ratios.append(save_peak_ratio)
        amplitude_distance.append(matched_amplitude_distance), frequency_distance.append(matched_frequency_distance)
    save_rmean = 1 - np.nanmean(save_ratios)

    results = {"save_rmean": save_rmean, "amplitude_distance": np.nanmean(amplitude_distance),
                 "frequency_distance": np.nanmean(frequency_distance)}

    return results
