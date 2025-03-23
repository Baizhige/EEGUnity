import os
import mne
import pandas as pd


def llm_boost_parser(file_path: str, client_type: str, client_paras: dict, completion_para: dict, max_iterations: int = 5):
    """
    Parses and processes an EEG data file using Azure OpenAI to generate a function
    that reads the data, calculates the sampling frequency, and extracts channel names.

    This function interacts with Azure OpenAI to automatically generate and refine a Python
    function that reads EEG data from a CSV or TXT file, determines the sampling frequency
    from timestamp columns, and extracts the relevant channel names. The function iterates
    through the process up to `max_iterations` times to refine the generated code in case
    of errors or unsatisfactory outputs.

    Parameters
    ----------
    file_path : str
        Path to the CSV or TXT file.
    client_type : str
        Type of LLM client to use (e.g., 'AzureOpenAI', 'OpenAI').
    client_paras : dict
        Parameters for initializing the LLM client.
    completion_para : dict
        Parameters for initializing the LLM completion process. Note: Parameter 'messages' is generated
        by this function, do not specify this parameter in 'completion_para'.
    max_iterations : (int, optional)
        Maximum number of iterations to refine the generated function code. Default is 5.

    Returns
    -------
        mne.io.Raw: An MNE RawArray object containing the processed EEG data.

    Raises
    ------
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the specified file is not found.
        RuntimeError: If the function code cannot be generated within the maximum iteration limit.

    Example
    -------
    >>> api_key = "your_api_key"
    >>> azure_endpoint = "https://your_endpoint"
    >>> locator_path = "data_file.csv"
    >>> raw_data = llm_boost_parser(locator_path, api_key, azure_endpoint)
    >>> print("Extracted Data:", raw_data)

    Contributor
    -----------
    Ziyi Jia (Ziyi.Jia21@student.xjtlu.edu.cn), on 2024-07-26.
    EEGUnity Team modified this file, on 2025-03-22.
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
    if client_type == "AzureOpenAI":
        from openai import AzureOpenAI
        client = AzureOpenAI(**client_paras)
    elif client_type == "OpenAI":
        from openai import OpenAI
        client = OpenAI(**client_paras)
    else:
        raise ValueError("Unsupported client_type. Supported types are 'AzureOpenAI' and 'OpenAI'.")

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
            **completion_para,
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
            if sfreq > 2000:
                raise ValueError(
                    f"The sampling rate now is {sfreq}, which is too large. Please revide your code. Make sure computation of sampling rate is right")
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
            raw_data = mne.io.RawArray(data, info)

            return raw_data

        except Exception as e:
            # Update the conversation history with the encountered error
            print(f"Error encountered: {e}")
            conversation_history = f"I have a CSV with description: \n {description}" + f"\n But there are some errors encountered: {e}\n The previous code was:\n{function_code} \n The expected returns should be 1. data: A ndarray with the shape (n_channels, n_times), containing the file data, without timestamp or string column. All columns must only contain float type data. \n 2. sfreq: A float representing the sampling frequency in Hz. \n 3. ch_names: A list of strings representing the channel names, must be same as columns name used in data. \n Please improve the code based on the above error and description." + f"\n Do not include any code block markers like ```python or other extra text. Return only the function code, without any additional text. This is program automatically request, the program will capture your code by function_code = response.choices[0].message.content.strip(), and employ it by exec(function_code, globals(), local_vars), data, sfreq, ch_names = local_vars['read_data'](file_path)"

    raise RuntimeError("Failed to generate valid code within the maximum iteration limit")
