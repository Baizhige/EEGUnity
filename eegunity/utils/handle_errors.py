import functools

def handle_errors(miss_bad_data: bool, error_list: list = None):
    """
    Decorator to handle errors in function execution based on the `miss_bad_data` flag.

    Parameters
    ----------
    miss_bad_data : bool
        If True, errors are caught and logged instead of raising exceptions.

    error_list : list, optional
        If provided, errors will be added to this list. Default is None (do not store errors).

    Returns
    -------
    Decorated function that handles errors as specified.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Assuming the first argument is `row` and contains a "File Path" key
                row = args[0] if args else None
                file_path = row["File Path"]

                if miss_bad_data:
                    print(f"Error: {e} (File: {file_path})")
                    if error_list is not None:
                        error_list.append(f"Error in file '{file_path}': {e}")  # Append the error with file path
                    return None
                else:
                    raise

        return wrapper

    return decorator