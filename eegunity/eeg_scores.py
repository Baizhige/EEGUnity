import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt

matplotlib.use('TkAgg')


# Design a Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Create a Butterworth bandpass filter.

    Parameters:
    lowcut : float
        The low frequency cut-off for the filter.
    highcut : float
        The high frequency cut-off for the filter.
    fs : int
        The sampling frequency of the data.
    order : int
        The order of the filter.

    Returns:
    b, a : ndarray, ndarray
        Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# Apply a bandpass filter to data
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to the data array.

    Parameters:
    data : ndarray
        Data to filter, with channels as rows.
    lowcut : float
        The low frequency cut-off for the filter.
    highcut : float
        The high frequency cut-off for the filter.
    fs : int
        The sampling frequency of the data.
    order : int
        The order of the filter.

    Returns:
    y : ndarray
        The filtered data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data, axis=1)  # Apply the filter along the second dimension (time)
    return y


# Plot a radar chart for EEG scores
def plot_radar_chart(scores, score_names, title='EEG Scores'):
    """
    Plot a radar chart for the given scores.

    Parameters:
    scores : list
        List of scores to plot.
    score_names : list
        Names corresponding to the scores.
    title : str
        Title for the radar chart.
    """
    angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    scores += scores[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.fill(angles, scores, color='lightseagreen', alpha=0.6)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylim(0, 100)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.set_title(title, size=20, color='darkslategray', y=1.1)

    for angle, score_name in zip(angles, score_names):
        ax.text(angle, 105, score_name, ha='center', va='center', fontsize=12)

    plt.show()


# Calculate general amplitude score across all channels
def calculate_general_amplitude_score(data):
    """
    Calculate a general amplitude score based on the proportion of signal amplitudes
    that fall within a specific range.

    Parameters:
    data : ndarray
        2D array of EEG data where rows represent channels and columns represent amplitude at each time point.

    Returns:
    float
        The average score across all channels, normalized to 100.
    """
    n_channels, n_samples = data.shape
    scores = np.zeros(n_channels)
    bins = np.arange(-1000, 1010, 10)

    for channel in range(n_channels):
        amplitudes = data[channel, :].flatten()
        hist, _ = np.histogram(amplitudes, bins=bins)
        normal_indices = (bins[:-1] >= -100) & (bins[:-1] <= 100)
        normal_sum = hist[normal_indices].sum()
        total_sum = hist.sum()
        scores[channel] = (normal_sum / total_sum) * 100 if total_sum > 0 else 0

    return np.mean(scores)


def calculate_highest_amplitude_score(data, channel_indices):
    """
    Calculate the highest amplitude score for specific channels within the Alpha band.

    Parameters:
    data : ndarray
        The EEG data in the Alpha band, expected to be a 2D array where rows represent channels and columns represent amplitude at each time point.
    channel_indices : list
        List of indices for the channels of interest.

    Returns:
    float
        The calculated highest amplitude score, normalized to 100.
    """
    # Filter out the channels of interest
    data_of_interest = data[channel_indices, :]

    # Get the maximum amplitude for each channel
    max_amplitudes = np.max(data_of_interest, axis=1)

    # Sorting indices by amplitude in descending order
    sorted_indices = np.argsort(-max_amplitudes)

    # Scoring: top 50% scores 1, next 25% scores 0.5, rest scores 0
    n_channels = len(max_amplitudes)
    scores = np.zeros(n_channels)
    top_50_percent_index = int(n_channels * 0.5)
    top_75_percent_index = int(n_channels * 0.75)

    scores[sorted_indices[:top_50_percent_index]] = 1
    scores[sorted_indices[top_50_percent_index:top_75_percent_index]] = 0.5
    # Lower 25% implicitly remains 0 as initialized

    # Calculate the final score
    final_score = np.mean(scores) * 100  # Normalize the score to 100

    return final_score


def calculate_dominant_frequency(signal, fs):
    """
    Calculate the dominant frequency of a signal.

    Parameters:
    signal : ndarray
        The signal array.
    fs : int
        The sampling frequency of the signal.

    Returns:
    float
        The dominant frequency of the signal.
    """
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1 / fs)[:n // 2]
    idx_max = np.argmax(np.abs(yf[:n // 2]))
    return xf[idx_max]


def calculate_symmetry_score(data, channels_left, channels_right, fs):
    """
    Calculate the symmetry score between two sets of channels.

    Parameters:
    data : ndarray
        The EEG data, expected to be a 2D array where rows represent channels and columns represent amplitude at each time point.
    channels_left : list
        List of indices for the left channels.
    channels_right : list
        List of indices for the right channels.
    fs : int
        The sampling frequency of the data.

    Returns:
    float
        The symmetry score between the two sets of channels.
    """
    # Filter the data for the left and right channels
    data_left = data[channels_left, :]
    data_right = data[channels_right, :]

    # Calculate the dominant frequencies for each set of channels
    dominant_frequencies_left = [calculate_dominant_frequency(data_left[ch], fs) for ch in range(len(channels_left))]
    dominant_frequencies_right = [calculate_dominant_frequency(data_right[ch], fs) for ch in range(len(channels_right))]

    # Calculate the correlation between the two sets of dominant frequencies
    correlation_score = np.abs(np.corrcoef(dominant_frequencies_left, dominant_frequencies_right)[0, 1]) * 100
    return correlation_score


def calculate_beta_sinusoidal_score(fft_data):
    """
    Calculate the beta sinusoidal score by analyzing the proportion of significant energy
    in the FFT data.

    Parameters:
    fft_data : ndarray
        The FFT results of the EEG data, expected to be a 2D array where each row represents the FFT results of a channel.

    Returns:
    float
        The sinusoidal score as a percentage.
    """
    total_energy = np.sum(np.abs(fft_data), axis=1)  # Total energy per channel
    max_energy = np.max(np.abs(fft_data), axis=1)  # Max energy per frequency per channel

    # Ensure threshold is properly broadcasted over the frequencies
    significant_threshold = 0.01 * max_energy[:, np.newaxis]  # Add new axis for broadcasting

    # Create a mask where the absolute FFT values are greater than the threshold
    significant_mask = np.abs(fft_data) > significant_threshold

    # Calculate significant energy using the mask
    significant_energy = np.sum(np.abs(fft_data) * significant_mask, axis=1)

    # Calculate the sinusoidal score for each channel
    significant_score = 100 * significant_energy / total_energy  # Convert to percentage
    significant_score[total_energy == 0] = 0  # Handle potential division by zero

    # Return the mean of significant scores across all channels
    return np.mean(significant_score)


def calculate_beta_amplitude_score(beta_data, threshold=20):
    """
    Calculate Score 4 based on the percentage of beta wave samples in each channel
    that do not exceed the maximum amplitude threshold.

    Parameters:
    beta_data : ndarray
        The EEG data filtered in the beta band, expected to be a 2D array
        where rows represent channels and columns represent amplitude at each time point.
    threshold : float
        The maximum amplitude threshold for the beta waves (in microvolts, μV).

    Returns:
    float
        The average percentage of samples across all channels not exceeding the threshold.
    """
    # Calculate the number of non-exceedance points per channel
    non_exceedances = np.abs(beta_data) <= threshold

    # Calculate the percentage of non-exceedances for each channel
    non_exceedance_percentage = np.sum(non_exceedances, axis=1) / beta_data.shape[1] * 100

    # Calculate the average score across all channels
    average_non_exceedance_percentage = np.mean(non_exceedance_percentage)

    return average_non_exceedance_percentage


def calculate_theta_amplitude_score(data, threshold=30):
    """
    Calculate the percentage of data points not exceeding a specified amplitude threshold in the theta frequency band.

    Parameters:
    data : ndarray
        The EEG data, expected to be a 2D array where rows represent channels and columns represent amplitude at each time point.
    threshold : float
        The amplitude threshold for considering a data point as not exceeding.

    Returns:
    float
        The average percentage of data points across all channels not exceeding the threshold.
    """
    # Check if data is already filtered for theta band or filter here if necessary
    # For example, if data is raw EEG, you would need to filter it for the theta band
    # using an appropriate band-pass filter.

    # Calculate non-exceedances
    non_exceedances = (data <= threshold).astype(int)  # Boolean array of where data does not exceed threshold
    non_exceedance_rate = np.mean(non_exceedances, axis=1)  # Mean non-exceedance rate per channel
    return np.mean(non_exceedance_rate) * 100  # Return average rate across all channels as a percentage


def classify_channels(channels):
    """
    Classify the given channels into different groups based on their location and type.

    Parameters:
    channels : list
        List of channel names.

    Returns:
    list
        List of channel indices that belong to Score 2.
    list
        List of left-side channel indices.
    list
        List of right-side channel indices.
    """
    # Define channel groups based on regions
    frontal = ['Fp', 'AF', 'F']
    temporal = ['FT', 'T', 'TP']
    parietal = ['C', 'CP', 'P']
    occipital = ['O', 'PO']
    auricular = ['A']  # Example for auricular electrodes like A1, A2

    score2_indices = []
    channels_left_indices = []
    channels_right_indices = []

    # Mapping to left or right based on odd/even convention in EEG naming
    for idx, ch in enumerate(channels):
        if any(ch.startswith(prefix) for prefix in frontal + temporal + parietal + occipital):
            if ch[-1] in '13579':  # Odd suffix typically denotes left side in standard systems
                channels_left_indices.append(idx)
            elif ch[-1] in '24680':  # Even suffix typically denotes right side
                channels_right_indices.append(idx)
            if ch.endswith('z') or ch.endswith('Z'):  # Midline channels are often included in broader score groups
                score2_indices.append(idx)

        if any(ch.startswith(prefix) for prefix in auricular):
            score2_indices.append(idx)  # Auricular often treated separately

    return score2_indices, channels_left_indices, channels_right_indices


def calculate_eeg_quality_scores(mne_io_raw):
    """
    Calculate EEG scores for the given data.

    Parameters:
    mne_io_raw : mne.io.Raw
        MNE Raw object containing EEG data.
    normalize : bool
        Flag indicating whether to normalize the data.

    Returns:
    list
        List of EEG scores corresponding to different metrics.
    """
    # unit convertion
    data = mne_io_raw.get_data()
    fs = int(mne_io_raw.info['sfreq'])  # Ensure fs is an integer
    channels = mne_io_raw.ch_names

    # Determine channel classification
    score2_indices, channels_left_indices, channels_right_indices = classify_channels(channels)

    # Filter data
    beta_data = butter_bandpass_filter(data, 12, 30, fs, order=4)
    theta_data = butter_bandpass_filter(data, 4, 8, fs, order=4)

    # Calculate scores
    general_amplitude_score = calculate_general_amplitude_score(data)
    beta_amplitude_score = calculate_beta_amplitude_score(beta_data)
    beta_sinusoidal_score = calculate_beta_sinusoidal_score(beta_data)
    theta_amplitude_score = calculate_theta_amplitude_score(theta_data)

    if (len(score2_indices) + len(channels_left_indices) + len(channels_right_indices) < len(
            channels)) or channels is None:
        scores = [
            general_amplitude_score,
            beta_amplitude_score,
            beta_sinusoidal_score,
            theta_amplitude_score
        ]

    else:
        # More channels provided, calculate additional scores
        alpha_data = butter_bandpass_filter(data, 8, 12, fs, order=4)
        highest_amplitude_score = calculate_highest_amplitude_score(alpha_data, score2_indices)
        dominant_frequency_score = calculate_symmetry_score(alpha_data, channels_left_indices, channels_right_indices,
                                                            fs)

        scores = [
            general_amplitude_score,
            highest_amplitude_score,
            dominant_frequency_score,
            beta_amplitude_score,
            beta_sinusoidal_score,
            theta_amplitude_score
        ]
    print(scores)

    return scores


if __name__ == "__main__":
    address = r'E:/test_dataset/test_sample_from_zenodo_saa.bdf'

    raw = mne.io.read_raw(address, preload=True)

    scores = calculate_eeg_quality_scores(raw)
    score_names = [
        "General Amplitude",
        "Highest Amplitude",
        "Dominant Frequency",
        "Beta Sinusoidal",
        "Beta Amplitude",
        "Theta Amplitude"
    ]
    plot_radar_chart(scores, score_names)
