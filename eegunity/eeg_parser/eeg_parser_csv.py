import pandas as pd
from datetime import datetime
import os
def calculate_interval(times):
    if isinstance(times.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype) or isinstance(times.iloc[0], pd.Timestamp):
        intervals = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
        average_interval = sum(intervals, pd.Timedelta(0)) / len(intervals) if intervals else pd.Timedelta(0)
        return average_interval.total_seconds()
    else:
        intervals = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
        return sum(intervals) / len(intervals) if intervals else None


def is_datetime_format(s):
    try:
        datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
        return True
    except ValueError:
        try:
            datetime.strptime(datetime.now().strftime("%Y-%m-%d ") + s, "%Y-%m-%d %H:%M:%S.%f")
            return True
        except ValueError:
            return False
def identify_time_columns(df):
    time_columns = {}
    for column in df.columns:
        if df[column].dtype == 'object' and all(df[column].apply(lambda x: ':' in str(x))):
            # Check if all entries are valid datetime or time strings
            if all(df[column].apply(is_datetime_format)):
                intervals = calculate_interval(pd.to_datetime(df[column], errors='coerce'))
                time_columns[column] = 1.0 / intervals if intervals else None

        elif df[column].dtype in ['float64', 'int'] and df[column].is_monotonic_increasing:
            # Check for monotonic increasing float series with small intervals
            intervals = calculate_interval(df[column])
            if intervals and intervals < 0.1:
                time_columns[column] = 1.0 / intervals
    if len(time_columns) == 1:
        key = list(time_columns.keys())[0]
        return key, time_columns[key]
    elif len(time_columns) > 1 and len(set(time_columns.values())) == 1:
        return list(time_columns.keys()), list(time_columns.values())[0]

    return None

def process_csv_files(files_locator):
    def calculate_sampling_rate(time_series):
        if not time_series.empty and not all(time_series == ''):
            time_series = pd.to_numeric(time_series, errors='coerce').dropna()
            if not time_series.empty:
                time_diffs = time_series.diff().dropna()
                average_sampling_interval = time_diffs.mean()
                if average_sampling_interval != 0:
                    return round(1 / average_sampling_interval)
                else:
                    return None
        return None

    for index, row in files_locator.iterrows():
        file_path = row['File Path']

        if (file_path.endswith('.csv') or file_path.endswith('.txt')) and os.path.getsize(
                file_path) > 5 * 1024 * 1024:
            print(file_path)
            try:
                # 判断文件的第一行是否为数字列名，决定是否需要header=None
                header_option = None if pd.read_csv(file_path, nrows=0).columns[0].isdigit() else 'infer'
                df = pd.read_csv(file_path, header=header_option)
                if header_option is None:
                    df.columns = [str(i) for i in range(1, len(df.columns) + 1)]

                files_locator.at[index, 'File Type'] = 'csvData'

                # Identifying timestamp columns
                time_info = identify_time_columns(df)

                if time_info is not None:
                    print(time_info)
                    time_cols = time_info[0]
                    files_locator.at[index, 'Sampling Rate'] = round(time_info[1])
                    # Recording channel names, excluding time column and non-float columns
                    channel_names = [col for col in df.columns if
                                     col not in time_cols and df[col].dtype in [float, 'float64']]
                    if channel_names:
                        files_locator.at[index, 'Channel Names'] = ','.join(channel_names)
                        files_locator.at[index, 'Number of Channels'] = len(channel_names)

                        data_shape = f"({len(channel_names)}, {len(df)})"
                        files_locator.at[index, 'Data Shape'] = data_shape

                        if 'Sampling Rate' in row and row['Sampling Rate'] != 'N.A.':
                            numeric_sampling_rate = pd.to_numeric(row['Sampling Rate'], errors='coerce')
                            if pd.notna(numeric_sampling_rate):
                                duration = len(df) / numeric_sampling_rate
                                files_locator.at[index, 'Duration'] = duration

            except pd.errors.ParserError:
                print(f"Failed to parse file as CSV: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    return files_locator
