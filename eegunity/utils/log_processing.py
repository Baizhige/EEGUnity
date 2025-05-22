from functools import wraps

def log_processing(func):
    """
    Decorator that logs the processing of a data row.

    This decorator prints a message indicating which row is being processed
    before calling the original function.

    Parameters
    ----------
    func : callable
        The function to decorate. It must accept a 'row' as its first argument.

    Returns
    -------
    callable
        The wrapped function with added logging behavior.
    """
    @wraps(func)
    def wrapper(row, *args, **kwargs):
        """
        Wrapper function that adds logging before calling the decorated function.
        """
        print(f"Processing row[{row}]")
        return func(row, *args, **kwargs)
    return wrapper