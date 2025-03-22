import csv
import os
import pandas as pd
from scipy.io import loadmat
import json
import re


def _read_files(directory):
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
                    if file.endswith('.txt') or file.endswith('.md'):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                        except UnicodeDecodeError:
                            with open(file_path, 'r', encoding='latin-1') as f:
                                file_content = f.read()
                        files.append({"file_path": file_path, "content": file_content})
                        print(f"read {file} complete, length：{len(file_content)}")
                    elif file.endswith('.docx'):
                        import docx
                        doc = docx.Document(file_path)
                        file_content = "\n".join([para.text for para in doc.paragraphs])
                        files.append({"file_path": file_path, "content": file_content})
                        print(f"read {file} complete")
                    elif file.endswith('.pdf'):
                        import pdfplumber
                        with pdfplumber.open(file_path) as pdf:
                            file_content = "\n".join([page.extract_text() or "" for page in pdf.pages])
                        files.append({"file_path": file_path, "content": file_content})
                        print(f"read {file} complete")
                    elif file.endswith('.csv'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            reader = csv.reader(f)
                            file_content = "\n".join([",".join(row) for row in reader])
                        files.append({"file_path": file_path, "content": file_content})
                        print(f"read {file} complete")
                    elif file.endswith('.xls') or file.endswith('.xlsx'):
                        df = pd.read_excel(file_path)
                        file_content = df.to_csv(index=False)
                        files.append({"file_path": file_path, "content": file_content})
                        print(f"read {file} complete")
                    elif file.endswith('.mat'):
                        try:
                            mat_data = loadmat(file_path)
                            file_content = str(mat_data)
                            files.append({"file_path": file_path, "content": file_content})
                            print(f"read {file} complete")
                        except Exception as e:
                            print(f"Error processing.mat file {file_path} :{e}")
                    elif 'readme' in file.lower() or 'annotation' in file.lower() or 'record' in file.lower():
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                        except UnicodeDecodeError:
                            with open(file_path, 'r', encoding='latin-1') as f:
                                file_content = f.read()
                        files.append({"file_path": file_path, "content": file_content})
                        print(f"read {file} complete, length：{len(file_content)}")

            except Exception as e:
                print(f"Error processing.mat file {file_path}：{e}")
    return files

def _filter_files_with_gpt(files: str, client_paras: dict, client_type: str, completion_para: dict):
    """
    Analyzes the given files using a GPT model and extracts JSON information containing
    sampling rate and channel names.

    Some large models may return mixed text; therefore, this function includes fault-tolerant
    logic to parse the response correctly.

    Args:
        files (List[Dict[str, Any]]): A list of file information dictionaries, each containing:
            - "file_path" (str): Path to the file.
            - "content" (str): File content.
        client_paras (Dict[str, Any]): Parameters required for initializing the GPT client.
        client_type (str): Specifies the type of GPT client to use. Supported types:
            - "AzureOpenAI"
            - "OpenAI"
        completion_para (Dict[str, Any]): Parameters for the GPT model completion request.

    Returns:
        List[Tuple[Dict[str, Any], Dict[str, Any]]]: A list of tuples where each tuple contains:
            - The original file information.
            - Extracted sampling rate and channel names in JSON format.
    """

    def parse_json_with_fallback(response_text: str):
        """
        Extracts and parses a JSON structure from the given GPT response text.

        The function attempts to retrieve a valid JSON object using the following steps:
        1. Prioritizes JSON code blocks enclosed by ```json ... ``` markers.
        2. If not found or parsing fails, attempts to extract JSON from the first occurrence
           of '{' to the last occurrence of '}'.
        3. If parsing still fails, raises a ValueError.

        Args:
            response_text (str): The text response from the GPT model.

        Returns:
            Dict[str, Any]: Extracted JSON data.

        Raises:
            ValueError: If no valid JSON can be extracted.
        """
        # Step 1: Attempt to extract JSON from ```json ... ``` blocks
        code_blocks = re.findall(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if code_blocks:
            for block in code_blocks:
                block = block.strip()
                try:
                    return json.loads(block)
                except Exception:
                    continue  # Try the next JSON block if parsing fails

        # Step 2: If no JSON block found, attempt to extract from the entire response
        text = response_text.strip()
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            possible_json = text[start_index:end_index + 1]
            try:
                return json.loads(possible_json)
            except Exception:
                pass

        # Step 3: If all attempts fail, raise an error
        raise ValueError("No valid JSON could be extracted from the GPT response.")
    processed_files = []
    try:
        if client_type == "AzureOpenAI":
            from openai import AzureOpenAI
            client = AzureOpenAI(**client_paras)
        elif client_type == "OpenAI":
            from openai import OpenAI
            client = OpenAI(**client_paras)
        else:
            raise ValueError("Unsupported client_type. Supported types are 'AzureOpenAI' and 'OpenAI'.")

        for file_info in files:
            try:
                file_path = file_info["file_path"]
                content = file_info["content"]
                print(f"Files being processed: {file_path}")

                response = client.chat.completions.create(
                    **completion_para,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                    "You are a highly capable assistant. Your task is to analyze the provided files and "
                                    "extract the **sampling rates** and **channel names** in a well-structured JSON format. "
                                    "The output should follow this structure:\n\n"
                                    "{\n"
                                    '    "sampling_rate": <number or null>,\n'
                                    '    "channels": [<channel_name_1>, <channel_name_2>, ...] or null\n'
                                    "}\n\n"
                                    "If either sampling rates or channel names are not available in the file, return `null` for the corresponding field. "
                                    "Ensure the response strictly follows the specified JSON format without additional text or explanations."                        ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps({
                                "file_path": file_path,
                                "content": content,
                                "format": "json"
                            })
                        }
                    ]
                )
                print(f"LLM response: {response}")
                response_text = response.choices[0].message.content
                decision = parse_json_with_fallback(response_text)
                processed_files.append((file_info, decision))
                print(f"Parsing LLM response: {decision}")
            except Exception as e:
                print(f"Error LLM response: {e}")
                continue

        return processed_files

    except Exception as e:
        print(f"Error using GPT to filter files: {e}")
        return processed_files


def _resolve_sampling_rate_conflict(sampling_rate_list):
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


def _resolve_channel_names_conflict(channel_info_list):
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


def llm_description_file_parser(directory: str, client_type: str, client_paras: dict, completion_para: dict):
    """
    Parse files in a specified directory to extract sampling rate and channel information using a Large Language Model
    (LLM) API.

    This function traverses a directory to read various file formats. It extracts sampling rates and channel names
    from the files using an LLM API (e.g., GPT-4), and processes the extracted information based on user inputs to
    resolve conflicts.

    Parameters
    ----------
    directory : str
        The directory path where the files are stored for processing. Generally speaking, it will be the root directory
        of the dataset
    client_type : str
        The type of LLM client to use (e.g., "AzureOpenAI", "OpenAI").
    client_paras : dict
        A dictionary containing the parameters needed to initialize the LLM API client. Please refer to OpenAI
        documentation.

    Returns
    -------

    dict: A dictionary containing the parsed sampling rate and channel information.
          Returns an error message if no files are selected or if all data is discarded due to conflicts.

    Raises
    ------
    ValueError: If no files are selected for further analysis or if there are conflicts in the extracted data.

    Examples
    --------
    >>> directory = 'path/to/description/directory'
    >>> client_paras = {"api_key": "your_api_key", "api_version": "2023-03-15-preview"}
    >>> client_type = "AzureOpenAI"
    >>> result = llm_description_file_parser(directory, client_paras, client_type)
    >>> print("The end result:", json.dumps(result, indent=4, ensure_ascii=False))

    Contributor
    -----------
    Jingyi Ding (Jingyi.Ding21@student.xjtlu.edu.cn), on 2024-07-26.
    EEGUnity Team modified it on 2025-02-23
    """
    files = _read_files(directory)
    processed_files = _filter_files_with_gpt(files, client_paras=client_paras, client_type=client_type, completion_para=completion_para)

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

    selected_sampling_rate, sampling_rate_msg = _resolve_sampling_rate_conflict(sampling_rate_list)
    if sampling_rate_msg:
        print(f" Sampling rate data: {sampling_rate_msg}")
        selected_info["sampling_rate"] = None
    else:
        selected_info["sampling_rate"] = selected_sampling_rate

    selected_channel_names, channel_names_msg = _resolve_channel_names_conflict(channel_info_list)
    if channel_names_msg:
        print(f" Channel name data: {channel_names_msg}")
        selected_info["channels"] = []
    else:
        selected_info["channels"] = selected_channel_names

    if selected_info["sampling_rate"] is None and not selected_info["channels"]:
        return {"error": "All data is discarded"}

    return {"sampling_rate": selected_info["sampling_rate"], "channels": selected_info["channels"]}
