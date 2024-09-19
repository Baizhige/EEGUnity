"""
Title: LLM Boost Parser using gpt4.0
Description: his project provides a Python script to parse and process EEG data files (CSV or TXT) using Azure OpenAI's GPT model to generate a function that reads the data, calculates the sampling frequency, and extracts channel names.
Author: Ziyi Jia
Email: Ziyi.Jia21@student.xjtlu.edu.cn
GitHub: [Ziyi-Jia05](https://github.com/1232353124)
Date Created: 2024-06-17
Last Modified: 2024-07-26
Version: 1.0
"""
# ----------------------------------------------------------------------
# import os
# import pandas as pd
# import mne
# from openai import AzureOpenAI


def llm_boost_parser(file_path: str, api_key: str, azure_endpoint: str, max_iterations: int = 5):
    """
    Parses and processes an EEG data file using Azure OpenAI to generate a function
    that reads the data, calculates the sampling frequency, and extracts channel names.

    Parameters:
        file_path (str): Path to the CSV or TXT file
        api_key (str): API key for Azure OpenAI.
        azure_endpoint (str): Endpoint URL for Azure OpenAI.
        max_iterations (int, optional): Maximum number of iterations to refine the generated function code. Default is 5.

    Returns:
        mne.io.Raw: An MNE RawArray object containing the processed EEG data.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the specified file is not found.
        RuntimeError: If the function code cannot be generated within the maximum iteration limit.
    Usage:
        api_key = ("your api key here")
        azure_endpoint = "https://your_endpoint"
        file_path = "data_file"
        raw_data = LLM_boost_parser(file_path, api_key, azure_endpoint)
        print("Extracted Data:")
        print(raw_data)
    """
    file_extension = os.path.splitext(file_path)[1]
    # Check if the file is a CSV or TXT file
    if file_extension == '.csv' or file_extension == '.txt':
        try:
            data = pd.read_csv(file_path)
            # Get the first ten rows for description
            first_ten_rows = data.head(10).to_string(index=False)
            columns = ', '.join(data.columns)
            description = f"CSV file with columns: {columns}. First ten rows:\n{first_ten_rows}"
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            return None
    else:
        raise ValueError("Unsupported file extension")

    client = AzureOpenAI(
        api_key=api_key,
        api_version="2023-03-15-preview",

        azure_endpoint=azure_endpoint
    )

    # Base prompt for generating the function code
    prompt_base = (
        f"Objective: Write a Python function with the following template:"
        f"def read_data(file_path: str):"
        f"    This function reads a CSV file (EEG data) from the specified file path and based on the description {description} returns three types of data: \n"
        f"    1. data: A ndarray with the shape (n_channels, n_times), containing the file data, without timestamp or string column. All columns must contain float type data"
        f"    2. sfreq: A float representing the sampling frequency in Hz."
        f"    3. ch_names: A list of strings representing the EEG channel names, must be same as columns name used in data."
        f"    The function should:"
        f"    - Read the CSV file into a pandas DataFrame."
        f"    - Automatically detect the 'Timestamp' column, which is likely to be contain string like 'Timestamp' or 'Time', without case sensitive. \n"
        f"    - Compute the sampling frequency (sfreq) based on the timestamp columns. For instance, try to calculate the difference between two nearby timestamp. The sampling rate ranges from 50Hz to 2000Hz\n"
        f"    - Return the data, sfreq, and ch_names."
        f"    For example, if the timestamps are ['2024-07-28 00:00:00', '2024-07-28 00:00:01', '2024-07-28 00:00:02'], the mean difference is 1 second, and the sampling frequency is 1 Hz. \n"
        f"    If the timestamps are ['2024-07-28 00:00:00.000', '2024-07-28 00:00:00.500', '2024-07-28 00:00:01.000'], the mean difference is 0.5 seconds, and the sampling frequency is 2 Hz. \n"
        f"    Do not include any code block markers like ```python or other extra text. Return only the function code, without any additional text. This is program automatical request, the program will capture your code by function_code = response.choices[0].message.content.strip(), and employ it by exec(function_code, globals(), local_vars), data, sfreq, ch_names = local_vars['read_data'](file_path)")
    conversation_history = prompt_base

    # Iterate to refine the function code if necessary
    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": conversation_history}
            ],
        )
        function_code = response.choices[0].message.content.strip()
        try:
            local_vars = {}
            exec(function_code, globals(), local_vars)
            data, sfreq, ch_names = local_vars['read_data'](file_path)
            if sfreq>2000:
                raise ValueError(f"The sampling rate now is {sfreq}, which is too large. Please revide your code. Make sure computation of sampling rate is right")
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
            raw_data = mne.io.RawArray(data, info)

            return raw_data

        except Exception as e:
            # Update the conversation history with the encountered error
            print(f"Error encountered: {e}")
            conversation_history = f"I have a CSV with description: \n {description}" + f"\n But there are some errors encountered: {e}\n The previous code was:\n{function_code} \n The expected returns should be 1. data: A ndarray with the shape (n_channels, n_times), containing the file data, without timestamp or string column. All columns must only contain float type data. \n 2. sfreq: A float representing the sampling frequency in Hz. \n 3. ch_names: A list of strings representing the channel names, must be same as columns name used in data. \n Please improve the code based on the above error and description."+f"\n Do not include any code block markers like ```python or other extra text. Return only the function code, without any additional text. This is program automatically request, the program will capture your code by function_code = response.choices[0].message.content.strip(), and employ it by exec(function_code, globals(), local_vars), data, sfreq, ch_names = local_vars['read_data'](file_path)"

    raise RuntimeError("Failed to generate valid code within the maximum iteration limit")