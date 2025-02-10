from functools import wraps

def log_processing(func):
    @wraps(func)
    def wrapper(row, *args, **kwargs):
        print(f"Processing row[{row}]")
        return func(row, *args, **kwargs)
    return wrapper