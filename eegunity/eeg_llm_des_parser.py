"""
Title: Multi-File Data Extraction and Analysis using gpt4.0
Description: This script traverses a specified directory to read various file formats,
             extract sampling rates and channel names using GPT4.0 model,
             and selected the expected information in the extracted data through user input.
Author: Jingyi Ding
Email: Jingyi.Ding21@student.xjtlu.edu.cn
Date Created: 2024-06-17
Last Modified: 2024-07-26
Version: 1.0
"""

import json
import csv
import os
import docx
import pdfplumber
import pandas as pd
from openai import AzureOpenAI
from scipy.io import loadmat


def read_files(directory):
    files = []
    print(f"Traversing the directory：{directory}")
    for root, dirs, files_in_dir in os.walk(directory):
        print(f"Current directory：{root}")
        for file in files_in_dir:
            file_path = os.path.join(root, file)
            print(f"Find file：{file_path}")
            try:
                size = os.path.getsize(file_path)
                print(f"File size：{size} byte")
                if size < 3 * 1024 * 1024:
                    if file_path.endswith('.txt') or file_path.endswith('.md'):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                        except UnicodeDecodeError:
                            with open(file_path, 'r', encoding='latin-1') as f:
                                file_content = f.read()
                        files.append({"file_path": file_path, "content": file_content})
                        print(f"read {file} complete, length：{len(file_content)}")
                    elif file_path.endswith('.docx'):
                        doc = docx.Document(file_path)
                        file_content = "\n".join([para.text for para in doc.paragraphs])
                        files.append({"file_path": file_path, "content": file_content})
                        print(f"read {file} complete")
                    elif file_path.endswith('.pdf'):
                        with pdfplumber.open(file_path) as pdf:
                            file_content = "\n".join([page.extract_text() or "" for page in pdf.pages])
                        files.append({"file_path": file_path, "content": file_content})
                        print(f"read {file} complete")
                    elif file_path.endswith('.csv'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            reader = csv.reader(f)
                            file_content = "\n".join([",".join(row) for row in reader])
                        files.append({"file_path": file_path, "content": file_content})
                        print(f"read {file} complete")
                    elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
                        df = pd.read_excel(file_path)
                        file_content = df.to_csv(index=False)
                        files.append({"file_path": file_path, "content": file_content})
                        print(f"read {file} complete")
                    elif file_path.endswith('.mat'):
                        try:
                            mat_data = loadmat(file_path)
                            file_content = str(mat_data)
                            files.append({"file_path": file_path, "content": file_content})
                            print(f"read {file} complete")
                        except Exception as e:
                            print(f"Error processing.mat file {file_path} :{e}")
            except Exception as e:
                print(f"Error processing.mat file {file_path}：{e}")
    return files


def filter_files_with_gpt(files, api_key, endpoint):
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2023-03-15-preview",
        azure_endpoint=endpoint,
    )

    try:
        processed_files = []
        for file_info in files:
            file_path = file_info["file_path"]
            content = file_info["content"]
            print(f"Files being processed：{file_path}")

            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant. Your task is to analyze the provided files and extract sampling rates and channel names in JSON format."},
                    {"role": "user",
                     "content": json.dumps({
                         "file_path": file_path,
                         "content": content,
                         "format": "json"
                     })}
                ]
            )
            print(f"GPT response：{response}")

            try:
                decision = json.loads(response.choices[0].message.content.strip())
                processed_files.append((file_info, decision))
            except Exception as e:
                print(f"Error parsing GPT response：{e}")

        return processed_files

    except Exception as e:
        print(f"Error using GPT to filter files：{e}")
        return []


def resolve_sampling_rate_conflict(sampling_rate_list):
    if not sampling_rate_list:
        return None, "Sample rate information not found"

    print("Multiple different sample rate information was detected：")
    for i, (file_info, sampling_rate) in enumerate(sampling_rate_list, 1):
        print(f"{i}: From file {file_info['file_path']}")
        print(f"   Sampling rate: {sampling_rate}")
    print(f"{len(sampling_rate_list) + 1}: Using no sample rate information")

    chosen_index = None
    while chosen_index not in range(1, len(sampling_rate_list) + 2):
        try:
            user_input = input(f"Please select a number for the sample rate information (input format is '1'）: ")
            chosen_index = int(user_input)
        except ValueError:
            print("Invalid selection, please re-enter.")

    if chosen_index == len(sampling_rate_list) + 1:
        return None, "Sample rate data is discarded"
    else:
        return sampling_rate_list[chosen_index - 1][1], None


def resolve_channel_names_conflict(channel_info_list):
    if not channel_info_list:
        return [], "No channel name information found"

    print("Several different channel name information was detected:")
    for i, (file_info, channel_names) in enumerate(channel_info_list, 1):
        print(f"{i}: From file {file_info['file_path']}")
        print(f"   Channel names: {channel_names}")
    print(f"{len(channel_info_list) + 1}: Using no channel names information")

    chosen_index = None
    while chosen_index not in range(1, len(channel_info_list) + 2):
        try:
            user_input = input(f"Please select a number for the channel name information (input format is '1'): ")
            chosen_index = int(user_input)
        except ValueError:
            print("Invalid selection, please re-enter.")

    if chosen_index == len(channel_info_list) + 1:
        return [], "The channel name data was discarded."
    else:
        return channel_info_list[chosen_index - 1][1], None


def llm_description_file_parser(api_key, endpoint, directory):
    """
    Parses files in a specified directory using an LLM (Large Language Model) API to identify and extract sampling rate
    and channel information.

    Parameters:
    -----------
    api_key : str
        The API key used to authenticate with the LLM service.
    endpoint : str
        The endpoint URL for the LLM API.
    directory : str
        The directory path where the files to be processed are located.

    Returns:
    --------
    dict
        A dictionary containing the parsed sampling rate and channel information. If no files are selected for further
        analysis or if all data is discarded due to conflicts, an error message is returned.

    Usage:
    ------
    api_key = "you_api_key"
    azure_endpoint = "https://your/end/point"
    directory = 'path/to/description/file'
    result = llm_description_file_parser(api_key, azure_endpoint, directory)
    print("The end result:", json.dumps(result, indent=4, ensure_ascii=False))
    """
    files = read_files(directory)
    processed_files = filter_files_with_gpt(files, api_key, endpoint)

    if not processed_files:
        return {"error": "No files were selected for further analysis"}

    sampling_rate_list = []
    channel_info_list = []
    channel_keys = ["channel_names", "channels", "channel name", "names of channel"]
    sampling_rate_keys = ["sampling_rate", "sampling rates", "sample rate", "samplingrate"]

    for file_info, decision in processed_files:
        for key in sampling_rate_keys:
            if key in decision:
                sampling_rate_list.append((file_info, decision[key]))
                break

        for key in channel_keys:
            if key in decision:
                channel_info_list.append((file_info, decision[key]))
                break

    selected_info = {"sampling_rate": None, "channels": []}

    selected_sampling_rate, sampling_rate_msg = resolve_sampling_rate_conflict(sampling_rate_list)
    if sampling_rate_msg:
        print(f" Sampling rate data: {sampling_rate_msg}")
        selected_info["sampling_rate"] = None
    else:
        selected_info["sampling_rate"] = selected_sampling_rate

    selected_channel_names, channel_names_msg = resolve_channel_names_conflict(channel_info_list)
    if channel_names_msg:
        print(f" Channel name data: {channel_names_msg}")
        selected_info["channels"] = []
    else:
        selected_info["channels"] = selected_channel_names

    if selected_info["sampling_rate"] is None and not selected_info["channels"]:
        return {"error": "All data is discarded"}

    return {"sampling_rate": selected_info["sampling_rate"], "channels": selected_info["channels"]}
