from eegunity.share_attributes import UDatasetSharedAttributes
import pandas as pd
import os
import copy
import mne
import numpy as np
from eegunity.eeg_parser import get_data_row, format_channel_names, extract_events, infer_channel_unit
from eegunity.eeg_scores import calculate_eeg_quality_scores
import warnings
import json
import ast

with open('combined_montage.json', 'r') as file:
    data = json.load(file)

# 获取键值并转化成列表
STANDARD_EEG_CHANNELS = list(data.keys())


class EEGBatch(UDatasetSharedAttributes):
    def __init__(self, main_instance):
        super().__init__()
        self._shared_attr = main_instance._shared_attr

    def batch_process(self, con_func, app_func, is_patch, result_type=None):
        """
        This function processes each row of the given dataframe `locator` based on the conditions
        specified in `con_func` and applies `app_func` accordingly. The function handles both
        list and dataframe return types and ensures the result aligns with the `locator`'s rows
        based on the `is_patch` flag.

        Parameters:
        con_func (function): A function that takes a row of `locator` and returns True or False
                             to determine if `app_func` should be applied to that row.
        app_func (function): A function that processes a row of `locator` and returns the result.
        is_patch (bool): If True, the returned list length or dataframe rows will match `locator`'s
                         row count, using placeholder elements as needed.
        result_type (str, optional): Specifies the expected return type of `app_func` results.
                                     Can be "series", "value", or None (case insensitive). Defaults to None.

        Returns:
        None or list or pd.DataFrame: The processed results, either as a list or dataframe, depending
                                      on the `result_type` and `app_func` return type and consistency.
                                      Returns None if result_type is None.
        """

        if result_type is not None:
            result_type = result_type.lower()
            if result_type not in ["series", "value", None]:
                raise ValueError("Invalid result_type. Must be 'series', 'value', or None.")

        results = []
        for index, row in self.get_shared_attr()['locator'].iterrows():
            if con_func(row):
                result = app_func(row)
            else:
                if is_patch and result_type == "series":
                    result = row
                else:
                    result = None

            results.append(result)

        if result_type == "series":
            # Combine results into a DataFrame if app_func returns Series
            combined_results = pd.concat([res for res in results if res is not None], axis=1).T
            combined_results.reset_index(drop=True, inplace=True)  # 重置索引并删除原来的索引列
            return combined_results
        elif result_type == "value":
            # Collect results into a list if app_func returns values
            if is_patch:
                return results
            else:
                return [res for res in results if res is not None]
        else:
            return None

    def set_column(self, col_name: str, value: list):
        """
        Set the specified column in the locator with the given list of values.

        Args:
        col_name (str): The name of the column to be set.
        value (list): The list of values to set in the column.

        Returns:
        None

        Raises:
        ValueError: If the length of the value list does not match the number of rows in the dataframe.
        TypeError: If the input types are not as expected.
        """
        # Check if locator is a DataFrame
        if not isinstance(self.get_shared_attr()['locator'], pd.DataFrame):
            raise TypeError("locator must be a pandas DataFrame")

        # Check if col_name is a string
        if not isinstance(col_name, str):
            raise TypeError("col_name must be a string")

        # Check if value is a list
        if not isinstance(value, list):
            raise TypeError("value must be a list")

        # Check if the length of value matches the number of rows in locator
        if len(value) != len(self.get_shared_attr()['locator']):
            raise ValueError("Length of value list must match the number of rows in the dataframe")

        # Set the column with the provided values
        self.get_shared_attr()['locator'][col_name] = value

    def sample_filter(
            self,
            channel_number=None,
            sampling_rate=None,
            duration=None,
            completeness_check=None,
            domain_tag=None,
            file_type=None
    ):
        """
        Filters the 'locator' dataframe based on the given criteria.

        Parameters:
        channel_number (tuple/list/array-like, optional): A tuple or list with (min, max) values to filter the
            "Number of Channels" column. If None, this criterion is ignored. Default is None.
        sampling_rate (tuple/list/array-like, optional): A tuple or list with (min, max) values to filter the
            "Sampling Rate" column. If None, this criterion is ignored. Default is None.
        duration (tuple/list/array-like, optional): A tuple or list with (min, max) values to filter the
            "Duration" column. If None, this criterion is ignored. Default is None.
        completeness_check (str, optional): A string that can be 'Completed', 'Unavailable', or 'Acceptable' to filter the
            "Completeness Check" column. The check is case-insensitive. If None, this criterion is ignored. Default is None.
        domain_tag (str, optional): A string to filter the "Domain Tag" column. If None, this criterion is ignored. Default is None.
        file_type (str, optional): A string to filter the "File Type" column. If None, this criterion is ignored. Default is None.

        Returns:
        None. The function updates the 'locator' dataframe in the shared attributes.
        """

        def con_func(row):
            """
            Condition function to determine if a row meets the filtering criteria.

            Parameters:
            row (pd.Series): A row from the 'locator' dataframe.

            Returns:
            bool: True if the row meets all the specified criteria, False otherwise.
            """
            if channel_number:
                min_channels, max_channels = channel_number
                if not (min_channels <= float(row["Number of Channels"]) <= max_channels):
                    return False
            if sampling_rate:
                min_rate, max_rate = sampling_rate
                if not (min_rate <= float(row["Sampling Rate"]) <= max_rate):
                    return False
            if duration:
                min_duration, max_duration = duration
                if not (min_duration <= float(row["Duration"]) <= max_duration):
                    return False
            if completeness_check:
                if row["Completeness Check"].strip().lower() != completeness_check.strip().lower():
                    return False
            if domain_tag:
                if row["Domain Tag"] != domain_tag:
                    return False
            if file_type:
                if row["File Type"] != file_type:
                    return False
            return True

        def app_func(row):
            """
            Apply function to process a row that meets the condition.

            Parameters:
            row (pd.Series): A row from the 'locator' dataframe.

            Returns:
            pd.Series: The same row if it meets the condition.
            """
            return row

        # Process the dataframe
        filtered_locator = self.batch_process(con_func, app_func, is_patch=False, result_type='series')

        # Update the locator in shared attributes
        self.set_shared_attr({'locator': filtered_locator})

    def save_as_other(self, output_path, domain_tag=None):
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output path does not exist: {output_path}")

        def con_func(row):
            return domain_tag is None or row['Domain Tag'] == domain_tag

        def app_func(row):
            raw = get_data_row(row)
            file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
            new_file_path = os.path.join(output_path, f"{file_name}_raw.fif")
            print(row['File Path'])
            raw.save(new_file_path, overwrite=True)
            row['File Path'] = new_file_path
            row['File Type'] = "standard_data"
            return row

        copied_instance = copy.deepcopy(self)
        new_locator = self.batch_process(con_func, app_func, is_patch=False, result_type='series')
        copied_instance.set_shared_attr({'locator': new_locator})
        return copied_instance

    def process_mean_std(self, domain_mean=True):
        def get_mean_std(data: mne.io.Raw):
            """
            Calculate mean and standard deviation for each channel and all channels in the given EEG data.

            Parameters:
            data (mne.io.Raw): The raw EEG data.

            Returns:
            dict: A dictionary containing the mean and standard deviation for all channels combined
                  and for each individual channel.
            """
            # Get data from Raw object
            data_array = data.get_data()

            # Get EEG channel indices
            eeg_channel_indices = mne.pick_types(data.info, eeg=True, meg=False, stim=False, eog=False)

            # Calculate mean and std for all EEG channels combined
            eeg_data = data_array[eeg_channel_indices, :]
            all_mean = np.mean(eeg_data)
            all_std = np.std(eeg_data)

            # Initialize result dictionary
            result = {"all_eeg": (all_mean, all_std)}

            # Calculate mean and std for each channel
            for i, ch_name in enumerate(data.ch_names):
                ch_mean = np.mean(data_array[i, :])
                ch_std = np.std(data_array[i, :])
                result[ch_name] = (ch_mean, ch_std)

            return result

        def con_func(row):
            return True

        def app_func(row):
            data = get_data_row(row)
            mean_std = get_mean_std(data)
            return mean_std

        # 使用 batch_process 处理数据
        results = self.batch_process(con_func, app_func, is_patch=False, result_type="value")

        if domain_mean:
            domain_results = {}
            for index, row in self.get_shared_attr()['locator'].iterrows():
                domain_tag = row['Domain Tag']
                if domain_tag not in domain_results:
                    domain_results[domain_tag] = []
                domain_results[domain_tag].append(results[index])

            # Check for channel consistency within each domain
            for domain_tag, res_list in domain_results.items():
                all_keys_sets = [set(res.keys()) - {'all'} for res in res_list]  # Exclude the 'all' key
                if len(set(map(frozenset, all_keys_sets))) != 1:
                    raise ValueError(
                        f"Inconsistent channel names in domain '{domain_tag}', please try domain_mean=False or check the locator manually and make sure all channel names are the same within a domain.")
                num_channels = [len(keys) for keys in all_keys_sets]
                if len(set(num_channels)) != 1:
                    raise ValueError(
                        f"Inconsistent number of channels in domain '{domain_tag}', please try domain_mean=False or check the locator manually and make sure all channel names are the same within a domain.")

            domain_mean_std = {}
            for domain_tag, res_list in domain_results.items():
                all_keys = res_list[0].keys()
                mean_std_dict = {}
                for key in all_keys:
                    mean_sum = np.mean([res[key][0] for res in res_list])
                    std_sum = np.mean([res[key][1] for res in res_list])
                    mean_std_dict[key] = (mean_sum, std_sum)
                domain_mean_std[domain_tag] = mean_std_dict

            new_results = []
            for index, row in self.get_shared_attr()['locator'].iterrows():
                domain_tag = row['Domain Tag']
                new_results.append(domain_mean_std[domain_tag])
            self.set_column("MEAN STD", new_results)
        else:
            self.set_column("MEAN STD", results)

    def format_channel_names(self):
        cache = {}

        def con_func(row):
            return True

        def app_func(row):
            channel_name = row['Channel Names']
            if channel_name in cache:
                return cache[channel_name]
            else:
                formatted_channel_name = format_channel_names(channel_name)
                cache[channel_name] = formatted_channel_name
                return formatted_channel_name

        # 使用 batch_process 处理数据
        results = self.batch_process(con_func, app_func, is_patch=False, result_type="value")

        self.set_column("Channel Names", results)

    def filter(self, output_path, filter_type='bandpass', l_freq=None, h_freq=None, notch_freq=None,
               auto_adjust_h_freq=True, picks='all', miss_bad_data=False):
        """
        对数据进行滤波，支持低通、高通、带通和陷波滤波器。

        参数：
        filter_type: 滤波类型，可以是 'lowpass', 'highpass', 'bandpass', 'notch'。
        l_freq: 滤波的低截止频率（高通滤波或带通滤波的低频段）。
        h_freq: 滤波的高截止频率（低通滤波或带通滤波的高频段）。
        notch_freq: 陷波滤波器的频率。
        output_path: 滤波后的文件输出路径。
        auto_adjust_h_freq: 是否自动调整高截止频率以适应奈奎斯特频率。
        picks: 涉及滤波使用的通道。
        miss_bad_data: 是否在处理出错时跳过当前文件继续处理下一个文件。
        """

        def con_func(row):
            return True

        def app_func(row, l_freq, h_freq, notch_freq, filter_type, output_path, auto_adjust_h_freq):
            try:
                # 生成路径
                file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                new_file_path = os.path.join(output_path, f"{file_name}_raw.fif")

                # 使用 get_data_row(row) 获取 mne.io.Raw 对象
                mne_raw = get_data_row(row)

                # 获取数据采样率
                sfreq = mne_raw.info['sfreq']
                nyquist_freq = sfreq / 2.0

                # 检查并自动调整高截止频率
                if h_freq is not None and h_freq >= nyquist_freq:
                    if auto_adjust_h_freq:
                        warnings.warn(
                            f"High-pass frequency ({h_freq} Hz) is greater than or equal to Nyquist frequency ({nyquist_freq} Hz). Adjusting h_freq to {nyquist_freq - 1} Hz.")
                        h_freq = nyquist_freq - 1
                    else:
                        raise ValueError(
                            f"High-pass frequency must be less than Nyquist frequency ({nyquist_freq} Hz).")

                # 根据滤波类型进行滤波
                if filter_type == 'lowpass':
                    mne_raw.filter(l_freq=None, h_freq=h_freq, fir_design='firwin', picks=picks)
                elif filter_type == 'highpass':
                    mne_raw.filter(l_freq=l_freq, h_freq=None, fir_design='firwin', picks=picks)
                elif filter_type == 'bandpass':
                    mne_raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', picks=picks)
                elif filter_type == 'notch':
                    if notch_freq is not None:
                        mne_raw.notch_filter(freqs=notch_freq, fir_design='firwin', picks=picks)
                    else:
                        raise ValueError("notch_freq must be specified for notch filter.")
                else:
                    raise ValueError("Invalid filter_type. Must be 'lowpass', 'highpass', 'bandpass', or 'notch'.")

                # 保存滤波后的数据
                mne_raw.save(new_file_path, overwrite=True)
                row['File Path'] = new_file_path

                return new_file_path
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return ""  # Return an empty path to indicate failure
                else:
                    raise

        # 使用 batch_process 处理数据
        new_path_list = self.batch_process(con_func,
                                           lambda row: app_func(row, l_freq, h_freq, notch_freq, filter_type,
                                                                output_path, auto_adjust_h_freq),
                                           is_patch=False,
                                           result_type='value')

        # 设置新的文件路径列，并移除路径为空的行
        self.set_column("File Path", new_path_list)
        locator_df = self.get_shared_attr()['locator']
        locator_df = locator_df[locator_df['File Path'] != ""]
        self.get_shared_attr()['locator'] = locator_df

    def ica(self, output_path, max_components=20, method='fastica', random_state=42, max_iter=1000, picks='eeg',
            miss_bad_data=False):
        """
        对数据进行ICA处理并去噪。

        参数：
        max_components: ICA组件的最大数量。
        method: ICA方法，可以是 'fastica', 'infomax', 'extended-infomax' 等。
        random_state: 随机状态，用于保证结果的可重复性。
        max_iter: ICA的最大迭代次数。
        output_path: 处理后的文件输出路径。
        picks: 涉及ICA处理的通道。
        miss_bad_data: 是否在处理出错时跳过当前文件继续处理下一个文件。
        """

        def con_func(row):
            return True

        def app_func(row, max_components, method, random_state, max_iter, output_path):
            try:
                # 生成路径
                file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                new_file_path = os.path.join(output_path, f"{file_name}_ica_cleaned_raw.fif")

                # 使用 get_data_row(row) 获取 mne.io.Raw 对象
                mne_raw = get_data_row(row)

                # 获取picks通道的数量
                picks_channels = mne.pick_types(mne_raw.info, meg=False, eeg=(picks == 'eeg'), eog=False)
                n_components = min(len(picks_channels), max_components)
                if n_components == 0:
                    print(f"No EEG channels in file {row['File Path']}")
                    return ""
                # 初始化ICA对象
                ica = mne.preprocessing.ICA(n_components=n_components, method=method, random_state=random_state,
                                            max_iter=max_iter)

                # 拟合ICA
                ica.fit(mne_raw, picks=picks)

                # 应用ICA到原始数据
                ica.apply(mne_raw)

                # 保存处理后的数据
                mne_raw.save(new_file_path, overwrite=True)
                row['File Path'] = new_file_path

                return new_file_path
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return ""  # Return an empty path to indicate failure
                else:
                    raise

        # 使用 batch_process 处理数据
        new_path_list = self.batch_process(con_func,
                                           lambda row: app_func(row, max_components, method, random_state, max_iter,
                                                                output_path),
                                           is_patch=False,
                                           result_type='value')

        # 设置新的文件路径列，并移除路径为空的行
        self.set_column("File Path", new_path_list)
        locator_df = self.get_shared_attr()['locator']
        locator_df = locator_df[locator_df['File Path'] != ""]
        self.get_shared_attr()['locator'] = locator_df

    def resample(self, output_path, new_sfreq, miss_bad_data=False):
        """
        对数据进行重采样。
        参数：
        output_path: 重采样后的文件输出路径。
        new_sfreq: 重采样后的新采样率。
        picks: 涉及重采样的通道。
        miss_bad_data: 是否在处理出错时跳过当前文件继续处理下一个文件。
        """

        def con_func(row):
            return True

        def app_func(row, new_sfreq, output_path):
            try:
                # 生成路径
                file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                new_file_path = os.path.join(output_path, f"{file_name}_resampled_raw.fif")

                # 使用 get_data_row(row) 获取 mne.io.Raw 对象
                mne_raw = get_data_row(row)

                # 对数据进行重采样
                mne_raw.resample(sfreq=new_sfreq)

                # 保存重采样后的数据
                mne_raw.save(new_file_path, overwrite=True)
                row['File Path'] = new_file_path
                row['Duration'] = str(
                    float(row['Sampling Rate'].strip()) * float(row['Duration'].strip()) / int(new_sfreq))
                row['Sampling Rate'] = str(int(new_sfreq))
                # 去掉括号并分割字符串
                dimensions = row['Data Shape'].strip('()').split(',')
                # 转换为整数列表
                dimensions = [int(dim.strip()) for dim in dimensions]
                dimensions[dimensions.index(max(dimensions))] = int(float(row['Duration'].strip()) * float(new_sfreq))
                # 将列表重新转换为字符串并添加括号
                row['Data Shape'] = f"({dimensions[0]}, {dimensions[1]})"
                return row
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return None  # Return an empty path to indicate failure
                else:
                    raise

        # 使用 batch_process 处理数据
        new_locator = self.batch_process(con_func,
                                         lambda row: app_func(row, new_sfreq, output_path),
                                         is_patch=False,
                                         result_type='series')
        self.get_shared_attr()['locator'] = new_locator

    def align_channel(self, output_path, channel_order, min_num_channels=32, miss_bad_data=False):
        """
        对数据进行通道顺序调整和插值。

        参数：
        output_path: 调整后文件输出路径。
        channel_order: 需要对齐的通道顺序，列表形式。
        min_num_channels: 最小匹配通道数。
        picks: 涉及的通道。
        miss_bad_data: 是否在处理出错时跳过当前文件继续处理下一个文件。
        """

        # 检查channel_order是否合法
        invalid_channels = [ch for ch in channel_order if ch not in STANDARD_EEG_CHANNELS]
        if invalid_channels:
            raise ValueError(f"Invalid channels found: {invalid_channels}")

        def con_func(row):
            return True

        def app_func(row, channel_order, output_path, min_num_channels):
            try:
                # 生成路径
                file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                new_file_path = os.path.join(output_path, f"{file_name}_aligned_raw.fif")

                # 使用 get_data_row(row) 获取 mne.io.Raw 对象
                mne_raw = get_data_row(row, norm_type="channel-wise")

                # 确保所有指定通道存在
                existing_channels = mne_raw.ch_names
                matched_channels = [ch for ch in channel_order if ch in existing_channels]

                # 检查匹配的通道数
                if len(matched_channels) < min_num_channels:
                    if miss_bad_data:
                        print(f"File {row['File Path']} skipped due to insufficient matching channels.")
                        return None  # Return an empty path to indicate failure
                    else:
                        raise ValueError(
                            f"File {row['File Path']} has insufficient matching channels. Required: {min_num_channels}, Found: {len(matched_channels)}")

                # 排列通道顺序并进行插值
                mne_raw.pick_channels(matched_channels, ordered=True)
                if len(matched_channels) < len(channel_order):
                    missing_channels = [ch for ch in channel_order if ch not in existing_channels]
                    mne_raw.add_channels(
                        [mne.create_info(missing_channels, sfreq=mne_raw.info['sfreq'], ch_types='eeg')])
                    mne_raw.interpolate_bads(reset_bads=False, origin='auto')

                # 保存调整后的数据
                mne_raw.save(new_file_path, overwrite=True)
                row['File Path'] = new_file_path
                row['Channel Names'] = ', '.join(mne_raw.info['ch_names'])
                row['Number of Channels'] = str(len(channel_order))
                # 去掉括号并分割字符串
                dimensions = row['Data Shape'].strip('()').split(',')
                # 转换为整数列表
                dimensions = [int(dim.strip()) for dim in dimensions]
                dimensions[dimensions.index(min(dimensions))] = len(channel_order)
                # 将列表重新转换为字符串并添加括号
                row['Data Shape'] = f"({dimensions[0]}, {dimensions[1]})"
                return row
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return None
                else:
                    raise

        # 使用 batch_process 处理数据
        new_locator = self.batch_process(con_func,
                                         lambda row: app_func(row, channel_order, output_path, min_num_channels),
                                         is_patch=False,
                                         result_type='series')
        self.get_shared_attr()['locator'] = new_locator

    def normalize(self, output_path, norm_type='sample-wise', miss_bad_data=False):
        """
        对数据进行重采样。
        参数：
        output_path: 归一化后的文件输出路径。
        norm_type：指定的归一化类型
        miss_bad_data: 是否在处理出错时跳过当前文件继续处理下一个文件。
        """

        def con_func(row):
            return True

        def app_func(row, norm_type, output_path):
            try:
                # 生成路径
                file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                new_file_path = os.path.join(output_path, f"{file_name}_normed_raw.fif")

                # 使用 get_data_row(row) 获取 mne.io.Raw 对象
                mne_raw = get_data_row(row, norm_type=norm_type)

                # 保存重采样后的数据
                mne_raw.save(new_file_path, overwrite=True)
                row['File Path'] = new_file_path
                row['File Type'] = "standard_data"
                return row
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return None  # Return an empty path to indicate failure
                else:
                    raise

        # 使用 batch_process 处理数据
        new_locator = self.batch_process(con_func,
                                         lambda row: app_func(row, norm_type, output_path),
                                         is_patch=False,
                                         result_type='series')
        self.get_shared_attr()['locator'] = new_locator

    def epoch_for_pretraining(self, output_path, seg_sec: float, resample: int = None, overlap: float = 0,
                              exclude_bad=True, baseline=(None, 0), miss_bad_data=False):
        def con_func(row):
            return True

        def app_func(row, output_path: str, seg_sec: float, resample: int = None, overlap: float = 0,
                     exclude_bad=True, baseline=(None, 0)):
            try:
                raw_data = get_data_row(row)
                if resample:
                    raw_data.resample(resample)

                # Calculate step size and event intervals
                step_sec = seg_sec * (1 - overlap)
                if baseline is not None and baseline[0] is not None and baseline[0] < 0:
                    start = -1 * baseline[0]
                else:
                    start = 0
                events = mne.make_fixed_length_events(raw_data, start=start, stop=None, duration=step_sec)
                event_id = {'segment': 1}

                # Create epochs
                epochs = mne.Epochs(raw_data, events, event_id, tmin=0, tmax=seg_sec,
                                    baseline=baseline, preload=True)

                # Exclude bad epochs
                if exclude_bad:
                    epochs.drop_bad()

                # Convert epochs to numpy array and Save
                file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                np.save(os.path.join(output_path, f"{file_name}_pretrain_epoch.npy"), epochs.get_data())
                return None
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return None  # Return an empty path to indicate failure
                else:
                    raise

        # 使用 batch_process 处理数据
        self.batch_process(con_func,
                           lambda row: app_func(row, output_path, seg_sec=seg_sec, resample=resample,
                                                overlap=overlap, exclude_bad=exclude_bad,
                                                baseline=baseline),
                           is_patch=False,
                           result_type=None)

    def get_events(self, miss_bad_data=False):
        """
        提取事件并记录在数据行中。

        参数：
        output_path: 提取事件后的文件输出路径。
        miss_bad_data: 是否在处理出错时跳过当前文件继续处理下一个文件。
        """

        def con_func(row):
            return True

        def app_func(row):
            try:
                # 生成路径
                # 使用 get_data_row(row) 获取 mne.io.Raw 对象
                mne_raw = get_data_row(row)

                # 提取事件和事件ID
                events, event_id = extract_events(mne_raw)

                # 将事件ID字典转化为字符串存储在row["event_id"]列中
                row["event_id"] = str(event_id)

                # 记录每一个事件的数量
                event_id_num = {key: sum(events[:, 2] == val) for key, val in event_id.items()}
                row["event_id_num"] = str(event_id_num)
                return row
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return None  # Return None to indicate failure
                else:
                    raise

        # 使用 batch_process 处理数据
        new_locator = self.batch_process(con_func,
                                         lambda row: app_func(row),
                                         is_patch=False,
                                         result_type='series')
        self.get_shared_attr()['locator'] = new_locator

    def epoch_by_event(self, output_path: str, seg_sec: float, resample: int = None,
                       exclude_bad=True, baseline=(0, 0.2), miss_bad_data=False):
        """
        Batch process EEG data to create epochs based on events specified in event_id column.

        Parameters:
        df (pd.DataFrame): DataFrame containing paths to raw EEG data and event_id information.
        output_path (str): Directory to save the processed epochs.
        seg_sec (float): Length of each epoch in seconds.
        resample (int): Resample rate for the raw data. If None, no resampling is performed.
        overlap (float): Fraction of overlap between consecutive epochs. Range is 0 to 1.
        exclude_bad (bool): Whether to exclude bad epochs. Uses simple heuristics to determine bad epochs.
        baseline (tuple): Time interval to use for baseline correction. If (None, 0), uses the pre-stimulus interval.
        miss_bad_data (bool): Whether to skip files with processing errors.

        Returns:
        None
        """

        def con_func(row):
            return True

        def app_func(row, output_path: str, seg_sec: float, resample: int = None,
                     exclude_bad=True, baseline=(None, 0), event_repeated="merge"):
            try:
                # Load raw data
                raw_data = get_data_row(row)

                # Resample if needed
                if resample:
                    raw_data.resample(resample)

                # Extract event_id from row
                event_id_str = row.get('event_id')
                # Convert event_id string to dictionary
                events, _ = extract_events(raw_data)
                event_id = ast.literal_eval(event_id_str)
                if not event_id_str or len(event_id)==0:
                    print(f"No event_id found for file {row['File Path']}")
                    return None
                # Create epochs
                epochs = mne.Epochs(raw_data, events, event_id, tmin=0, tmax=seg_sec,
                                    baseline=baseline, preload=True, event_repeated=event_repeated)
                # Exclude bad epochs
                if exclude_bad:
                    epochs.drop_bad()

                # Convert epochs to numpy array and save to respective subdirectories
                for event in event_id:
                    event_epochs = epochs[event]
                    file_name = os.path.splitext(os.path.basename(row['File Path']))[0]
                    event_output_path = os.path.join(output_path, event)
                    os.makedirs(event_output_path, exist_ok=True)
                    np.save(os.path.join(event_output_path, f"{file_name}_{event}_epoch.npy"), event_epochs.get_data())

                return None
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return None
                else:
                    raise

        # Use batch_process to process data
        self.batch_process(con_func,
                           app_func=lambda row: app_func(row, output_path, seg_sec=seg_sec, resample=resample, exclude_bad=exclude_bad, baseline=baseline),
                           is_patch=False,
                           result_type=None)

    def infer_units(self, miss_bad_data=False):
        """
        推断每个通道的单位并记录在数据行中。

        参数：
        miss_bad_data: 是否在处理出错时跳过当前文件继续处理下一个文件。
        """

        def con_func(row):
            return True

        def app_func(row):
            try:
                # 生成路径
                # 使用 get_data_row(row) 获取 mne.io.Raw 对象
                mne_raw = get_data_row(row)

                # 初始化单位字典
                units = {}

                # 获取通道信息
                ch_names = mne_raw.info['ch_names']
                ch_types = mne_raw.get_channel_types()

                for ch_name, ch_type in zip(ch_names, ch_types):
                    # 获取通道数据
                    ch_data = mne_raw.get_data(picks=[ch_name])

                    # 使用静态函数推断单位
                    unit = infer_channel_unit(ch_name, ch_data, ch_type)
                    units[ch_name] = unit

                # 将单位字典转化为字符串存储在row["Infer Unit"]列中
                row["Infer Unit"] = str(units)
                return row
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return None  # Return None to indicate failure
                else:
                    raise

        # 使用 batch_process 处理数据
        new_locator = self.batch_process(con_func,
                                         lambda row: app_func(row),
                                         is_patch=False,
                                         result_type='series')
        self.get_shared_attr()['locator'] = new_locator

    def get_quality(self, miss_bad_data=False):

        def con_func(row):
            return True

        def app_func(row):
            try:
                raw_data = get_data_row(row, unit_convert='uV')
                scores = calculate_eeg_quality_scores(raw_data)
                score = np.mean(scores)
                return str(score)
            except Exception as e:
                if miss_bad_data:
                    print(f"Error processing file {row['File Path']}: {e}")
                    return ""  # Return None to indicate failure
                else:
                    raise
        # 使用 batch_process 处理数据
        results = self.batch_process(con_func, app_func, is_patch=False, result_type="value")

        self.set_column("Score", results)