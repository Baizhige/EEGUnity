import os
import pprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from math import ceil, sqrt
from eegunity.modules.parser.eeg_parser import get_data_row
from eegunity._share_attributes import _UDatasetSharedAttributes


class EEGCorrection(_UDatasetSharedAttributes):
    def __init__(self, main_instance):
        super().__init__()
        self._shared_attr = main_instance._shared_attr

    def report(self):
        """
        Generate a statistical report on the dataset, providing proportions for file types,
        sampling rates, channel configurations, and completeness checks. The report can be
        generated for the dataset as a whole or for individual groups based on domain tags.

        The report includes:
        - File Type Proportions (%)
        - Sampling Rate Proportions (%)
        - Channel Configuration Proportions (%)
        - Completeness Check Proportions (%)

        Returns
        -------
            dict: A dictionary containing the computed proportions for file types,
            sampling rates, channel configurations, and completeness checks for each domain or overall.

        Examples
        --------
        >>> unified_dataset.eeg_correction.report()
        """

        def percentage(part, whole):
            return round(100 * float(part) / float(whole), 2)

        def generate_statistics(grouped_df, overall=False):
            result = {}
            domain_tags = ['Overall'] if overall else grouped_df.groups.keys()

            for domain_tag in domain_tags:
                if overall:
                    data = self.get_shared_attr()['locator']
                else:
                    data = grouped_df.get_group(domain_tag)

                total_count = len(data)
                # Calculate the percentage of each file type in the dataset/group
                file_type_counts = {k: percentage(v, total_count) for k, v in
                                    data['File Type'].value_counts().to_dict().items()}
                # Calculate the percentage of each sampling rate in the dataset/group
                sampling_rate_counts = {k: percentage(v, total_count) for k, v in
                                        data['Sampling Rate'].value_counts().to_dict().items()}
                # Calculate the percentage of each channel configuration in the dataset/group
                channel_configs = data['Channel Names'].apply(
                    lambda x: f'configuration {len(x.split(","))}' if pd.notna(x) else 'unknown').value_counts()
                channel_configs_percentage = {k: percentage(v, total_count) for k, v in
                                              channel_configs.to_dict().items()}
                # Calculate the percentage of each completeness check status in the dataset/group
                completeness_check_counts = {k: percentage(v, total_count) for k, v in
                                             data['Completeness Check'].value_counts().to_dict().items()}
                # Store the calculated statistics in the result dictionary
                result[domain_tag] = {
                    'File Type Proportions (%)': file_type_counts,
                    'Sampling Rate Proportions (%)': sampling_rate_counts,
                    'Configuration Proportions (%)': channel_configs_percentage,
                    'Completeness Check Proportions (%)': completeness_check_counts,
                }

            return result

        def generate_diagnostics(grouped_df):
            result = {}
            for domain_tag, data in grouped_df:
                channel_names = data['Channel Names']
                formatted_channels = channel_names.apply(
                    lambda x: all([":" in ch for ch in str(x).split(",")]) if pd.notna(x) else False)
                formatted_count = formatted_channels.sum()

                unknown_channels = data.loc[formatted_channels, 'Channel Names'].apply(
                    lambda x: [ch for ch in str(x).split(",") if 'unknown' in ch.lower()])

                incomplete_files = data[data['Completeness Check'] != 'Completed']['File Path'].tolist()

                result[domain_tag] = {
                    'Formatted Channel Names Count': formatted_count,
                    'Unknown Channels': unknown_channels.to_dict(),
                    'Incomplete Files': incomplete_files,
                }

            return result

        grouped = self.get_shared_attr()['locator'].groupby('Domain Tag')

        statistics = generate_statistics(grouped)
        overall_statistics = generate_statistics(grouped, overall=True)
        diagnostics = generate_diagnostics(grouped)

        # Combine all statistics and diagnostics in a single report
        report = {
            'Statistics': statistics,
            'Overall Statistics': overall_statistics,
            'Diagnostics': diagnostics,
        }
        pprint.pprint(report)

    def visualization_frequency(self, max_sample:int = 10):
        """
        Visualize the frequency spectrum for each domain in the dataset.

        For each domain in the dataset, this function computes and visualizes the
        amplitude spectrum of the data. The number of samples visualized per domain
        is limited by the `max_sample` parameter. If a domain contains more samples
        than this limit, a random subset will be selected.

        Parameters
        ----------
        max_sample : int, optional
            The maximum number of samples to visualize per domain. If the number of
            samples in a domain exceeds this value, a random subset will be used.
            The default value is 10.

        Returns
        -------
        None
            This function does not return any value. It displays frequency spectrum
            plots for each domain in the dataset.

        Note
        ----
        If a domain contains inconsistent sampling rates across its samples, the
        data will be resampled to the lowest sampling rate within the domain.
        Frequency bands for visualization are fixed as follows: 0-4 Hz, 4-8 Hz,
        and 8-13 Hz, 13-30 Hz.

        Example
        -------
        >>> unified_dataset.eeg_batch.visualization_frequency(max_sample=5)
        >>> # This will visualize the frequency spectrum for up to 5 samples per domain from the dataset.
        """
        def compute_amplitude_spectrum(data, sfreq):
            # Compute amplitude spectrum using scipy's welch method
            freqs, psd = welch(data, float(sfreq), window='hann', nperseg=1024, noverlap=512, nfft=2048, axis=-1)
            return psd, freqs

        locator = self.get_shared_attr()['locator']
        domains = locator['Domain Tag'].unique()

        for i, domain in enumerate(domains):
            domain_data = locator[locator['Domain Tag'] == domain]
            if domain_data.shape[0] > max_sample:
                domain_data = domain_data.sample(n=max_sample)

            sample_rates = domain_data['Sampling Rate'].dropna().unique()
            if len(sample_rates) > 1:
                print(f"Warning: Inconsistent sampling rates in domain {domain}. Resampling to the lowest rate.")
                new_sfreq = min(sample_rates)
            else:
                new_sfreq = sample_rates[0]

            all_spectra = []
            for _, row in domain_data.iterrows():
                mne_data = get_data_row(row)
                if mne_data.info['sfreq'] != new_sfreq:
                    mne_data.resample(new_sfreq)

                data = mne_data.get_data()
                data = data.astype(np.float64)  # Ensure data is of type float64
                psd, freqs = compute_amplitude_spectrum(data, new_sfreq)
                all_spectra.append(psd.mean(axis=0))

            all_spectra = np.array(all_spectra)
            mean_spectrum = all_spectra.mean(axis=0)

            # Define frequency bands
            bands = [(0, 4), (4, 8), (8, 13), (13, 30)]
            band_names = ['0-4 Hz', '4-8 Hz', '8-13 Hz', '13-30 Hz']

            fig, axes = plt.subplots(4, 1, figsize=(10, 15))
            for ax, (low, high), band_name in zip(axes, bands, band_names):
                idx = np.where((freqs >= low) & (freqs <= high))
                for spectrum in all_spectra:
                    ax.plot(freqs[idx], spectrum[idx], color='lightgray', alpha=0.7)
                ax.plot(freqs[idx], mean_spectrum[idx], color='blue', linewidth=2)
                ax.set_title(f"Domain: {domain} - {band_name}")
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Amplitude')

                # Set major and minor ticks
                ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Adjusted to avoid too many ticks
                ax.xaxis.set_minor_locator(plt.MaxNLocator(20))  # Adjusted to avoid too many ticks
                ax.yaxis.set_major_locator(plt.MaxNLocator(10))  # Adjusted to avoid too many ticks
                ax.yaxis.set_minor_locator(plt.MaxNLocator(20))  # Adjusted to avoid too many ticks

                # Add grid lines for major ticks
                ax.grid(which='both')
                ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
                ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

            plt.tight_layout()
            plt.show()

    def visualization_channels_corr(self, max_sample=16):
        """
        Visualize the correlation between EEG data channels for different domains.

        This method computes and visualizes the correlation matrix between EEG channels for each domain
        available in the dataset. For each domain, a sample of EEG data is selected, and the correlation
        between channels is plotted as a matrix.

        Parameters
        ----------
        max_sample : int, optional
            The maximum number of samples to visualize per domain. If the number of available samples
            exceeds this value, a random subset of samples is selected. Default is 16.

        Returns
        -------
        None
            This function does not return any value. It generates and displays a set of correlation matrix
            plots for each domain.

        Note
        ----
        - This function uses the `np.corrcoef` method to compute the correlation matrix.
        - The visualization is created using `matplotlib`, and the correlation values range from -1 to 1.

        Example
        -------
        >>> # To visualize the correlation matrices for up to 10 samples per domain:
        >>> unified_dataset.eeg_batch.visualization_channels_corr(max_sample=10)
        """

        def compute_channel_correlation(data):
            # Compute correlation matrix for EEG data channels
            corr_matrix = np.corrcoef(data)
            return corr_matrix

        locator = self.get_shared_attr()['locator']
        domains = locator['Domain Tag'].unique()

        for i, domain in enumerate(domains):
            domain_data = locator[locator['Domain Tag'] == domain]
            if domain_data.shape[0] > max_sample:
                domain_data = domain_data.sample(n=max_sample)

            n = ceil(sqrt(max_sample))
            fig, axes = plt.subplots(n, n, figsize=(15, 15))
            axes = axes.flatten()

            for j, (_, row) in enumerate(domain_data.iterrows()):
                if j >= max_sample:
                    break
                mne_data = get_data_row(row)
                data = mne_data.get_data()
                corr_matrix = compute_channel_correlation(data)

                ax = axes[j]
                cax = ax.matshow(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
                fig.colorbar(cax, ax=ax)
                file_name = os.path.basename(row['File Path'])
                ax.set_title(f'{file_name}')
                ax.set_xticks([])
                ax.set_yticks([])

            # Remove any unused subplots
            for k in range(j + 1, len(axes)):
                fig.delaxes(axes[k])

            plt.suptitle(f'Channel Correlation for Domain: {domain}')
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()
