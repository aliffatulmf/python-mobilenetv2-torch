def run_once(func):
    """
    A decorator that runs a function only once.
    """
    has_run = False

    def wrapper(*args, **kwargs):
        nonlocal has_run
        if not has_run:
            result = func(*args, **kwargs)
            has_run = True
            return result
        else:
            return None
    return wrapper
