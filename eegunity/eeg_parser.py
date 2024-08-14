import os
import pandas as pd
import glob
import mne
import zipfile
import re
import scipy
from eegunity.share_attributes import UDatasetSharedAttributes
from eegunity.eeg_parser_mat import process_mat_files, _find_variables_by_condition, _condition_source_data
from eegunity.eeg_parser_csv import process_csv_files
import ast
import warnings
import json
import datetime
import numpy as np
with open('combined_montage.json', 'r') as file:
    data = json.load(file)

# 获取键值并转化成列表
STANDARD_EEG_CHANNELS = list(data.keys())

EEG_PREFIXES_SUFFIXES = {"EEG", "FP", "REF", "LE", "RE"}

class EEGParser(UDatasetSharedAttributes):
    def __init__(self, main_instance):
        super().__init__()
        self._shared_attr = main_instance._shared_attr
        self.format_channel_names = format_channel_names
        dataset_path = self.get_shared_attr()['dataset_path']
        locator_path = self.get_shared_attr()['locator_path']
        if dataset_path and locator_path:
            raise ValueError("不能同时存在'datasets'和'locator'路径")
        elif not dataset_path and not locator_path:
            raise ValueError("必须提供'datasets'或'locator'路径之一")

        if self.get_shared_attr()['locator_path']:  # 通过读取Locator地址构建UnifiedDataset
            if os.path.isfile(locator_path) and locator_path.endswith('.csv'):
                print("已从现有CSV加载数据：")
                self.locator_path = locator_path
                self.set_shared_attr({'locator': self.check_locator(pd.read_csv(locator_path))})
            else:
                raise ValueError("提供的'locator'路径不是有效的CSV文件")
        elif self.get_shared_attr()['dataset_path']:  # 通过读取数据集地址构建UnifiedDataset
            if os.path.isdir(dataset_path):
                self._unzip_if_no_conflict(dataset_path)
                self.set_shared_attr({'locator': self.check_locator(self._process_directory(dataset_path))})
            else:
                raise ValueError("提供的'datasets'路径不是有效的目录")

    def _process_directory(self, datasets_path):
        # 递归搜索目录下的所有文件，直接更新self.files_locator
        files_info = []
        for filepath in glob.glob(datasets_path + '/**/*', recursive=True):
            if os.path.isfile(filepath):
                files_info.append([filepath, self.get_shared_attr()['domain_tag'], '', '', '', '', '', '', ''])
        files_locator = pd.DataFrame(files_info,
                                     columns=['File Path', 'Domain Tag', 'File Type', 'Data Shape', 'Channel Names',
                                              'Number of Channels',
                                              'Sampling Rate', 'Duration', 'Completeness Check'])
        files_locator = process_mne_files(files_locator, self.get_shared_attr()['verbose'])
        files_locator = process_mat_files(files_locator)
        files_locator = process_csv_files(files_locator)
        files_locator = _clean_sampling_rate_(files_locator)
        return files_locator

    def _unzip_if_no_conflict(self, datasets_path):
        if self.get_shared_attr()['is_unzip']:
            print("已开启解压文件检索。当检测到zip文件时，会改变目录结构")
            # 递归遍历目录中的所有文件和子目录
            for root, dirs, files in os.walk(datasets_path):
                for filename in files:
                    # 检查文件是否是zip文件
                    if filename.endswith('.zip'):
                        file_path = os.path.join(root, filename)
                        # 检查zip文件解压后是否有同名文件存在
                        # 通常是检查去掉.zip后缀的文件名是否存在
                        if not os.path.exists(os.path.splitext(file_path)[0]):
                            # 没有同名文件，解压zip文件
                            try:
                                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                                    zip_ref.extractall(root)
                                    print(f"Extracted: {file_path}")
                            except zipfile.BadZipFile:
                                print(f"Bad zip file: {file_path}")
                            except Exception as e:
                                print(f"Failed to extract {file_path}: {e}")
                        else:
                            print(f"Skipped {file_path}, conflict exists.")
        else:
            return

    def get_data(self, data_idx, norm_type=None, unit_convert=False):
        row = self.get_shared_attr()['locator'].iloc[data_idx]
        return get_data_row(row, norm_type, self.get_shared_attr()['verbose'], unit_convert=unit_convert)

    def check_locator(self, locator):
        locator = locator.astype(str)
        def check_data_shape(data_shape):
            if data_shape.strip() == '':
                return ["Miss data in Data Shape"]
            try:
                dimensions = [int(dim) for dim in data_shape.strip('()').split(',')]
                if len(dimensions) != 2:
                    return ["Raw Data Shape is not two-dimensional"]
            except ValueError:
                return ["Data Shape format error"]
            return []

        def check_channel_duplicates(channel_names):
            if channel_names.strip() == '':
                return ["Miss data in Channel Names"]
            if channel_names.strip() == 'nan':
                return ["channel name is nan"]
            if pd.isna(channel_names):
                return ["channel name is nan"]
            if pd.isnull(channel_names):
                return ["channel name is null"]

            channel_names = channel_names.split(',')
            seen = set()
            duplicates = set()
            for name in channel_names:
                if name in seen:
                    duplicates.add(name)
                else:
                    seen.add(name)
            if duplicates:
                return [f"Duplicate channels: {', '.join(duplicates)}"]
            return []

        def check_channel_counts(data_shape, num_channels):
            try:
                dimensions = [int(dim) for dim in data_shape.strip('()').split(',')]
                if min(dimensions) != int(float(num_channels)):
                    return ["Mismatch in reported and actual channel counts"]
            except ValueError:
                return ["Channel count or data shape format error"]
            return []

        def check_duration(sampling_rate, duration, data_shape):
            try:
                dimensions = [int(dim) for dim in data_shape.strip('()').split(',')]
                calculated_duration = max(dimensions) / float(sampling_rate)
                if abs(calculated_duration - float(duration)) >= 1:
                    return ["Incorrect duration calculation"]
            except ValueError:
                return ["Duration or sampling rate format error"]
            return []

        # 遍历DataFrame的每一行
        for index, row in locator.iterrows():
            errors = []
            errors.extend(check_data_shape(row.get('Data Shape', '').strip()))

            if row.get('File Type', '').strip() == '':
                errors.append("Miss data in File Type")
            elif row['File Type'] == "unidentified":
                errors.append("File Type is 'unidentified'")

            if row.get('File Path', '').strip() == '':
                errors.append("Miss data in File Path")
            elif not os.path.exists(row['File Path']):
                errors.append("File does not exist")

            errors.extend(
                check_channel_counts(row.get('Data Shape', '').strip(), row.get('Number of Channels', '').strip()))
            errors.extend(check_duration(row.get('Sampling Rate', '').strip(), row.get('Duration', '').strip(),
                                         row.get('Data Shape', '').strip()))
            errors.extend(check_channel_duplicates(row.get('Channel Names', '').strip()))
            # 更新Check列
            locator.at[index, 'Completeness Check'] = "Completed" if not errors else "Unavailable"
        return locator


# static function defining
def normalize_data(raw_data, mean_std_str, norm_type):
    mean_std_str = mean_std_str.replace('nan', 'None')
    mean_std_dict = ast.literal_eval(mean_std_str)
    # 获取数据数组
    data = raw_data.get_data()
    channel_names = raw_data.info['ch_names']
    if norm_type == "channel-wise":
        for idx, channel in enumerate(channel_names):
            if channel in mean_std_dict:
                mean, std = mean_std_dict[channel]
                data[idx] = (data[idx] - mean) / std

    elif norm_type == "sample-wise":
        mean, std = mean_std_dict['all_eeg']
        for idx in range(data.shape[0]):
            data[idx] = (data[idx] - mean) / std

    # 将标准化后的数据写回 Raw 对象
    raw_data._data = data
    return raw_data


def set_montage_any(raw_data: mne.io.Raw, verbose='CRITICAL'):
    montage = create_montage_from_json('combined_montage.json')
    raw_data.set_montage(montage, on_missing= 'warn', verbose=verbose)
    return raw_data


def create_montage_from_json(json_file):
    with open(json_file, 'r') as f:
        montage_data = json.load(f)

    ch_names = list(montage_data.keys())
    pos = [montage_data[ch_name] for ch_name in ch_names]

    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, pos)))

    return montage

def set_channel_type(raw_data, channel_str):
    channel_info = [ch.split(':') for ch in channel_str.split(',')]
    # 确保每个分割后的元素长度为2
    for ch in channel_info:
        if len(ch) != 2:
            raise ValueError(f"Invalid channel format: {ch}. Each channel must be in 'type:name' format.")

    channel_mapping = {raw_channel: new_channel for raw_channel, new_channel in
                       zip(raw_data.info['ch_names'], [ch[1] for ch in channel_info])}
    raw_data.rename_channels(channel_mapping)
    for ch, ch_info in zip(raw_data.info['chs'], channel_info):
        if ch_info[0].strip() == 'EEG':
            ch['kind'] = mne.io.constants.FIFF.FIFFV_EEG_CH
        elif ch_info[0].strip() == 'EMG':
            ch['kind'] = mne.io.constants.FIFF.FIFFV_EMG_CH
        elif ch_info[0].strip() == 'ECG':
            ch['kind'] = mne.io.constants.FIFF.FIFFV_ECG_CH
        elif ch_info[0].strip() == 'EOG':
            ch['kind'] = mne.io.constants.FIFF.FIFFV_EOG_CH
        else:
            ch['kind'] = mne.io.constants.FIFF.FIFFV_BIO_CH
    return raw_data

def get_data_row(row, norm_type=None, is_set_channel_type=True, is_set_montage=True, verbose='CRITICAL', pick_types=None, unit_convert=None):
    filepath = row['File Path']
    file_type = row['File Type']
    # 读取mne.io.raw数据
    if file_type == "standard_data": # 读取标准EEG数据
        raw_data = mne.io.read_raw(filepath, verbose=verbose, preload=True)
        channel_names = [name.strip() for name in row['Channel Names'].split(',')]
        if len(channel_names) != len(raw_data.info['ch_names']):
            raise ValueError(f"locator文件所标记通道数量与元数据通道数量不一致{filepath}")
        channel_mapping = {original: new for original, new in zip(raw_data.info['ch_names'], channel_names)}
        raw_data.rename_channels(channel_mapping)
    else: # 处理非标准数据
        raw_data = handle_nonstandard_data(row, verbose)
    # 根据locator重设通道名称与类型
    if is_set_channel_type:
        raw_data = set_channel_type(raw_data, row['Channel Names'])
    # 根据预设monatge，设置电极坐标
    if is_set_montage:
        raw_data = set_montage_any(raw_data)
    # 归一化处理
    if norm_type and 'MEAN STD' in row:
        raw_data = normalize_data(raw_data, row['MEAN STD'], norm_type)
    # 重新设置通道单位
    if unit_convert and 'Infer Unit' in row:
        raw_data = set_infer_unit(raw_data, row)
        raw_data = convert_unit(raw_data, unit_convert)

    # 重建戳重设
    if raw_data.info['meas_date'] is not None and isinstance(raw_data.info['meas_date'],
                                                            datetime.datetime) and (
            raw_data.info['meas_date'].timestamp() < -2147483648 or raw_data.info[
        'meas_date'].timestamp() > 2147483647):
        # If the date is wrong, set None.
        raw_data.set_meas_date(None)

    return raw_data

def set_infer_unit(raw_data, row):
    infer_unit = ast.literal_eval(row['Infer Unit'])
    if isinstance(infer_unit, dict):
        for ch_name, unit in infer_unit.items():
            if ch_name in raw_data.info['ch_names']:
                idx = raw_data.ch_names.index(ch_name)
                raw_data.info['chs'][idx]['eegunity_unit'] = unit
        return raw_data
    else:
        raise ValueError(f"'Infer Unit' is not a valid dictionary: {row['Infer Unit']}")
def format_channel_names(input_string):

    # 定义检查是否为EOG通道的函数
    def is_eog_channel(channel):
        return "eog" in channel.lower()

    # 定义检查是否为MEG通道的函数
    def is_meg_channel(channel):
        return "meg" in channel.lower()

    # 定义检查是否为ECG通道的函数
    def is_ecg_channel(channel):
        return "ecg" in channel.lower()

    # 定义检查是否为双联通道的函数
    def is_double_channel(channel):
        if '-' not in channel:
            return False
        parts = channel.split('-')
        if len(parts) != 2:
            return False

        valid_channels = {"REF", "LE", "EEG", "ECG", "EOG", "EMG"}

        # Convert both parts to upper case for case-insensitive comparison
        part1, part2 = parts[0].upper(), parts[1].upper()

        if part1 in valid_channels or part2 in valid_channels:
            return False
        return True

    # 定义标准化EEG通道名称的函数
    def standardize_eeg_channel(channel):
        # 移除前缀
        channel = channel.replace('EEG:', '').replace('EEG', '').replace('eeg', '')
        return f"EEG:{channel}"

    def remap_standard_name(channel):
        # 移除前缀
        channel = channel.replace('EEG:', '').replace('EEG', '').replace('eeg', '')

        # 定义替换规则
        replacements = {
            'FAF': 'AFF',
            'CFC': 'FCC',
            'CPC': 'CCP',
            'POP': 'PPO',
            'TPT': 'TTP',
            'TFT': 'FTT'
        }
        # 执行替换并提出警告
        for old, new in replacements.items():
            if old.lower() in channel.lower():
                warnings.warn(
                    f'{old.upper()} is an invalid 10-5 name and has been replaced with {new.upper()}. \n If mismatch happen, you should change locator manually.')
                channel = channel.lower().replace(old.lower(), new.lower())
        return channel

    # 定义预处理通道名称的函数，去除前后空白及EEG相关前后缀
    def preprocess_channel(channel):
        channel = channel.replace(" ", "")
        for prefix_suffix in EEG_PREFIXES_SUFFIXES:
            if channel.startswith(prefix_suffix + "-"):
                channel = channel[len(prefix_suffix + "-"):].strip()
            if channel.startswith(prefix_suffix + ":"):
                channel = channel[len(prefix_suffix + ":"):].strip()
            if channel.startswith(prefix_suffix):
                channel = channel[len(prefix_suffix):].strip()
            if channel.endswith("-" + prefix_suffix):
                channel = channel[:-len("-" + prefix_suffix)].strip()
            if channel.endswith(prefix_suffix):
                channel = channel[:-len(prefix_suffix)].strip()
        return channel

    # 拆分输入字符串为通道名称列表
    channels = input_string.split(',')

    # 初始化格式化后的通道名称集合和已见通道名称集合
    formatted_channels = []
    seen_channels = set()

    # 处理每个通道名称
    for channel in channels:
        channel = preprocess_channel(channel)

        if is_eog_channel(channel):
            formatted_channel = f"EOG:{channel.replace('EOG:', '').replace('EOG', '').replace('eog', '')}"
        elif is_meg_channel(channel):
            formatted_channel = f"MEG:{channel.replace('MEG:', '').replace('MEG', '').replace('meg', '')}"
        elif is_ecg_channel(channel):
            formatted_channel = f"ECG:{channel.replace('ECG:', '').replace('ECG', '').replace('ecg', '')}"
        elif is_double_channel(channel):
            formatted_channel = f"EEGDual:{channel.replace('Dual:', '')}"
        else:
            re_channel = remap_standard_name(channel)
            matched = False
            for standard_name in STANDARD_EEG_CHANNELS:
                if re_channel.lower() == standard_name.lower():
                    formatted_channel = standardize_eeg_channel(standard_name)
                    matched = True
                    break
            if not matched:
                formatted_channel = f"Unknown:{channel.replace('Unknown:', '')}"
        # 检查是否重复
        if formatted_channel in seen_channels:
            warnings.warn(f"Duplicate formatted channel detected: {formatted_channel}")
            return input_string

        formatted_channels.append(formatted_channel)
        seen_channels.add(formatted_channel)

    # 拼接格式化后的通道名称为字符串
    output_string = ', '.join(formatted_channels)
    return output_string
def _clean_sampling_rate_(df):

    # 将Sampling Rate列转换为字符串，并保留小数点和科学记数法中的'e'
    df['Sampling Rate'] = df['Sampling Rate'].astype(str).apply(lambda x: re.sub(r'[^0-9.eE+-]', '', x))
    # 如果需要，可以将结果转回数值类型
    df['Sampling Rate'] = pd.to_numeric(df['Sampling Rate'], errors='coerce')

    return df

def handle_nonstandard_data(row, verbose='CRITICAL'):
    filepath = row['File Path']
    if filepath.endswith('.mat'):
        matdata = scipy.io.loadmat(filepath)
        eeg_data = \
            _find_variables_by_condition(matdata, _condition_source_data, max_depth=5, max_width=20)[1]
        if eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T

        channel_names = row['Channel Names'].split(',')
        info = mne.create_info(ch_names=channel_names, sfreq=float(row['Sampling Rate']), ch_types='eeg',verbose=verbose)
        raw = mne.io.RawArray(eeg_data, info)
        return raw

    elif (filepath.endswith('.csv') or filepath.endswith('.txt')) and row['File Type'] == 'csvData':
        # 从 locator 获取 header 信息
        # 检查是否有 header
        header_option = None if row.get('Header', 'infer') == 'None' else 'infer'
        df = pd.read_csv(filepath, header=header_option)

        if header_option is None:
            # 如果文件没有列名，按照索引生成列名
            df.columns = [str(i) for i in range(1, len(df.columns) + 1)]

        # 获取通道名和采样率
        channel_names = row['Channel Names'].split(',')
        sfreq = float(row['Sampling Rate'])

        # 检查所有通道名是否都在 DataFrame 的列中
        if not all(name in df.columns for name in channel_names):
            raise(f"locator文件所标记通道数量与元数据通道不一致{filepath}")
        # 提取 EEG 数据
        eeg_data = df[channel_names].values.T  # 转置以匹配 MNE 需要的格式 (通道数, 时间点数)
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info)
        return raw

    else:
        raise("当前不支持 .mat/.csv/.txt以外文件的解析")


def extract_events(raw):
    """
    从 mne.io.Raw 对象中提取事件。

    尝试使用 mne.find_events 提取事件，
    如果失败则使用 mne.events_from_annotations 提取事件。

    参数:
    raw : mne.io.Raw
        原始数据对象

    返回:
    events : numpy.ndarray
        事件数组，形状为 (n_events, 3)
    event_id : dict
        事件ID字典
    """
    try:
        # 尝试使用 mne.find_events 提取事件
        events = mne.find_events(raw)
        # 自动生成 event_id 字典，假设所有事件都是有效的
        unique_event_ids = np.unique(events[:, 2])
        event_id = {f'event_{event_id}': event_id for event_id in unique_event_ids}
        print("Events extracted using mne.find_events.")
    except ValueError as e:
        print(f"find_events failed with error: {e}")
        print("Trying to extract events from annotations.")
        # 使用 mne.events_from_annotations 提取事件
        events, event_id = mne.events_from_annotations(raw)
        print("Events extracted using mne.events_from_annotations.")

    return events, event_id


def infer_channel_unit(ch_name, ch_data, ch_type):
    """
    推断通道单位类型。

    参数：
    ch_name: 通道名称
    ch_data: 通道数据
    ch_type: 通道类型

    返回值：
    推断的单位类型，例如"uV"，"mV"，"V"等。
    """
    mean_val = abs(ch_data).mean()

    # 根据通道类型和平均幅值推断单位
    if ch_type == 'eeg' or ch_type == 'eog':
        if mean_val > 1:
            return "uV"
        elif mean_val > 0.001:
            return "mV"
        else:
            return "V"
    elif ch_type == 'ecg' or ch_type == 'emg':
        if mean_val > 1:
            return "uV"
        elif mean_val > 0.001:
            return "mV"
        else:
            return "V"
    else:
        # 对于misc和其他未知类型
        if mean_val > 1:
            return "uV"
        elif mean_val > 0.001:
            return "mV"
        else:
            return "V"


def convert_unit(data: mne.io.Raw, unit: str) -> mne.io.Raw:
    # 校验单位
    valid_units = ['V', 'mV', 'uV']
    if unit not in valid_units:
        raise ValueError(f"Invalid unit '{unit}'. Valid units are 'V', 'mV', 'uV'.")

    # 获取通道数
    n_channels = len(data.info['chs'])

    # 定义单位换算关系
    unit_conversion = {'V': 1, 'mV': 1e-3, 'uV': 1e-6}
    target_multiplier = unit_conversion[unit]

    # 遍历所有通道
    for i in range(n_channels):
        ch = data.info['chs'][i]
        current_unit = ch['eegunity_unit']

        if current_unit in unit_conversion:
            current_multiplier = unit_conversion[current_unit]
            conversion_factor = current_multiplier / target_multiplier
            # 换算数据
            data._data[i] *= conversion_factor

            # 更新单位
            ch['eegunity'] = unit

        # 标记已经进行过单位换算
        data.info['chs'][i].update({"eegunity_unit_converted": unit})

    return data

def process_mne_files(files_locator, verbose):
    for index, row in files_locator.iterrows():
        filepath = row['File Path']
        try:
            data = mne.io.read_raw(filepath, verbose=verbose)
            files_locator.at[index, 'File Type'] = 'standard_data'
            files_locator.at[index, 'Data Shape'] = str((data.info['nchan'], data.n_times))
            files_locator.at[index, 'Channel Names'] = ', '.join(data.info['ch_names'])
            files_locator.at[index, 'Number of Channels'] = len(data.info['ch_names'])
            files_locator.at[index, 'Sampling Rate'] = data.info['sfreq']
            files_locator.at[index, 'Duration'] = data.times[-1]
            print(f"检索到通道序列 {data.info['ch_names']}")
        except Exception:
            files_locator.at[index, 'File Type'] = 'unknown'
    return files_locator