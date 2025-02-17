import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Design a Butterworth bandpass filter.

    Parameters
    ----------
    low_cut : float
        The lower cutoff frequency of the filter.
    high_cut : float
        The upper cutoff frequency of the filter.
    sampling_freq : float
        The sampling frequency of the input signal.
    order : int, optional
        The order of the Butterworth filter (default is 4).

    Returns
    -------
    b : ndarray
        The numerator (b) coefficients of the IIR filter.
    a : ndarray
        The denominator (a) coefficients of the IIR filter.

    Note
    ----
    This function designs a bandpass filter using a Butterworth design.
    The cutoff frequencies are normalized by the Nyquist frequency,
    which is half the sampling frequency. The filter coefficients are
    returned as arrays suitable for use with `scipy.signal.lfilter` or
    similar filtering functions.
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

    Parameters
    ----------
    data : ndarray
        Data to filter, with channels as rows.
    lowcut : float
        The low frequency cut-off for the filter.
    highcut : float
        The high frequency cut-off for the filter.
    fs : int
        The sampling frequency of the data.
    order : int, optional
        The order of the filter (default is 4).

    Returns
    -------
    y : ndarray
        The filtered data.

    Note
    ----
    This function uses a Butterworth bandpass filter designed with the
    specified cut-off frequencies and order. The filter is applied using
    zero-phase filtering with `scipy.signal.filtfilt`, which ensures
    that the filtered signal is not phase-shifted.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data, axis=1)  # Apply the filter along the second dimension (time)
    return y


# Plot a radar chart for EEG scores
def plot_radar_chart(scores, score_names, title='EEG Scores'):
    """
    Plot a radar chart for the given scores.

    Parameters
    ----------
    scores : list of float
        List of scores to plot. Each score represents a metric for evaluating EEG quality.
    score_names : list of str
        Names corresponding to the scores. These names describe the metrics for evaluating EEG quality.
    title : str, optional
        Title for the radar chart (default is 'EEG Scores').

    Returns
    -------
    None

    Note
    ----
    This function creates a radar chart using the provided scores and score names.
    The radar chart is displayed using matplotlib.
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

    Parameters
    ----------
    data : ndarray
        2D array of EEG data where rows represent channels and columns represent amplitudes at each time point.

    Returns
    -------
    float
        The average score across all channels, normalized to 100.

    Note
    ----
    This function calculates the general amplitude score for each channel by counting the number of amplitudes
    within a specific range (-100 to 100) and dividing it by the total number of amplitudes. The scores are then
    averaged across all channels to obtain the final general amplitude score.
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

    Parameters
    ----------
    data : ndarray
        The EEG data in the Alpha band, expected to be a 2D array where rows represent channels and columns represent amplitudes at each time point.
    channel_indices : list of int
        List of indices for the channels of interest.

    Returns
    -------
    float
        The calculated highest amplitude score, normalized to 100.

    Note
    ----
    This function first filters the data for the specified channels of interest.
    It then calculates the maximum amplitude for each channel. The amplitudes are sorted
    in descending order, and scoring is applied based on the percentile of each channel's amplitude.
    The final score is the average of these scores, normalized to 100.
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

    Parameters
    ----------
    signal : ndarray
        The signal array. This should be a 1D array representing the time-series data of the EEG signal.
    fs : int
        The sampling frequency of the signal, representing the number of data points collected per second.

    Returns
    -------
    float
        The dominant frequency of the signal, which is the frequency component that has the highest amplitude in the
        Fourier Transform of the signal.

    Note
    ----
    This function computes the Fast Fourier Transform (FFT) of the input signal and identifies
    the frequency with the highest amplitude. The dominant frequency is determined from the
    positive frequency components of the FFT.
    """
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1 / fs)[:n // 2]
    idx_max = np.argmax(np.abs(yf[:n // 2]))
    return xf[idx_max]


def calculate_symmetry_score(data, channels_left, channels_right, fs):
    """
    Calculate the symmetry score between two sets of channels.

    Parameters
    ----------
    data : ndarray
        The EEG data, expected to be a 2D array where rows represent channels and columns represent amplitudes at each
        time point.
    channels_left : list of int
        List of indices for the left channels.
    channels_right : list of int
        List of indices for the right channels.
    fs : int
        The sampling frequency of the data.

    Returns
    -------
    float
        The symmetry score between the two sets of channels, expressed as a percentage.

    Note
    ----
    This function filters the data for the specified left and right channels, calculates the dominant frequencies
    for each set of channels, and computes the correlation score between the dominant frequencies. The symmetry
    score is normalized to a range of 0 to 100.
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

    Parameters
    ----------
    fft_data : ndarray
        The FFT results of the EEG data, expected to be a 2D array where each row represents the FFT results of a
        channel.

    Returns
    -------
    float
        The sinusoidal score as a percentage.

    Note
    ----
    This function computes the total and significant energy for each channel based on the FFT results.
    It applies a threshold to determine significant frequencies and calculates the score as the
    percentage of significant energy relative to the total energy. If the total energy is zero,
    the score for that channel is set to zero to avoid division errors.
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

    Parameters
    ----------
    beta_data : ndarray
        The EEG data filtered in the beta band, expected to be a 2D array
        where rows represent channels and columns represent amplitudes at each time point.
    threshold : float, optional
        The maximum amplitude threshold for the beta waves (in microvolts, μV). Default is 20.

    Returns
    -------
    float
        The average percentage of samples across all channels that do not exceed the threshold.

    Note
    ----
    This function counts the number of samples in each channel that are less than or equal
    to the specified threshold and calculates the percentage of such samples relative to the
    total number of samples in that channel. The final score is the average percentage across
    all channels.
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

    Parameters
    ----------
    data : ndarray
        The EEG data, expected to be a 2D array where rows represent channels and columns represent amplitudes at each
        time point.
    threshold : float, optional
        The amplitude threshold for considering a data point as not exceeding. Default is 30.

    Returns
    -------
    float
        The average percentage of data points across all channels that do not exceed the threshold.

    Note
    ----
    This function checks for data points in the theta frequency band that are below the specified threshold
    and calculates the percentage of such points for each channel. It returns the average percentage across
    all channels. If the input data is not already filtered for the theta band, appropriate filtering should
    be applied before using this function.
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

    Parameters
    ----------
    channels : list
        List of channel names.

    Returns
    -------
    list
        List of channel indices that belong to Score 2.
    list
        List of left-side channel indices.
    list
        List of right-side channel indices.

    Note
    ----
    This function classifies channels based on common EEG electrode naming conventions.
    Channels are grouped into frontal, temporal, parietal, occipital, and auricular categories.
    It distinguishes between left-side and right-side channels based on the suffix of their names,
    where odd-numbered suffixes typically denote left-side channels and even-numbered suffixes
    denote right-side channels. Midline and auricular channels are included in the Score 2 classification.
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


def compute_quality_scores_shady(mne_io_raw):
    """
    Calculate EEG scores for the given data.

    Parameters
    ----------
    mne_io_raw : mne.io.Raw
        MNE Raw object containing EEG data.

    Returns
    -------
    list
        List of EEG scores corresponding to different metrics.

    Note
    ----
    This function calculates various EEG quality scores based on different metrics.
    The scores are calculated using filtered data from the original EEG signals.
    If fewer channels are available, only selected scores are computed.
    Otherwise, additional metrics are included for a more comprehensive assessment.
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

    return scores
