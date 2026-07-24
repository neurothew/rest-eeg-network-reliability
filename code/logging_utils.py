"""Logging utility module for your project."""

import logging
import functools
import inspect
from contextlib import contextmanager
import sys
from typing import Optional, Union, Any, Callable
from pathlib import Path
from datetime import datetime

# Create the logger
logger = logging.getLogger("EEGfeat")
logger.propagate = False

# Logging levels mapping
LOGGING_TYPES = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

def setup_logging(default_level: str = 'INFO') -> None:
    """Initialize logging configuration.
    
    Parameters
    ----------
    default_level : str
        Default logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setFormatter(logging.Formatter('%(message)s'))
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    )
    logger.addHandler(console_handler)
    
    # Set initial logging level
    set_log_level(default_level)

def set_log_level(verbose: Union[str, bool, int, None] = None, 
                 return_old_level: bool = False) -> Optional[int]:
    """Set the logging level.
    
    Parameters
    ----------
    verbose : str, bool, int, or None
        The verbosity of messages to print:
        - If str: 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'
        - If bool: True is 'INFO', False is 'WARNING'
        - If None: defaults to 'INFO'
    return_old_level : bool
        If True, return the old verbosity level.
    
    Returns
    -------
    old_level : int, optional
        The old level (only if return_old_level is True)
    """
    old_verbose = logger.level
    
    # Parse verbose parameter
    if verbose is None:
        verbose = 'INFO'
    elif isinstance(verbose, bool):
        verbose = 'INFO' if verbose else 'WARNING'
    
    if isinstance(verbose, str):
        verbose = verbose.upper()
        if verbose not in LOGGING_TYPES:
            raise ValueError(f"verbose must be one of {list(LOGGING_TYPES.keys())}")
        verbose = LOGGING_TYPES[verbose]
    
    # Set the level
    logger.setLevel(verbose)
    
    return old_verbose if return_old_level else None


def _parse_log_level(verbose: Union[str, bool, int, None]) -> int:
    """Normalize a logging-level input without changing the logger."""
    if verbose is None:
        verbose = 'INFO'
    elif isinstance(verbose, bool):
        verbose = 'INFO' if verbose else 'WARNING'

    if isinstance(verbose, str):
        verbose = verbose.upper()
        if verbose not in LOGGING_TYPES:
            raise ValueError(f"verbose must be one of {list(LOGGING_TYPES.keys())}")
        verbose = LOGGING_TYPES[verbose]

    return verbose

# def set_log_file(filename: Union[str, Path], 
#                  mode: str = 'w',
#                  format: str = '%(asctime)s - %(levelname)s - %(message)s') -> None:
#     """Add a file handler to the logger.
    
#     Parameters
#     ----------
#     filename : str or Path
#         Path to the log file
#     mode : str
#         'w' to overwrite, 'a' to append
#     format : str
#         Format string for the log messages
#     """
#     file_handler = logging.FileHandler(filename, mode=mode)
#     file_handler.setFormatter(logging.Formatter(format))
#     logger.addHandler(file_handler)

def set_log_file(filename: Union[str, Path],
                 mode: str = 'w',
                 format: str = '%(asctime)s - %(levelname)s - %(message)s') -> Path:
    """Add a file handler to the logger.
    
    The filename will be appended with the current timestamp.
    
    Parameters
    ----------
    filename : str or Path
        Path to the log file (without a timestamp).
    mode : str
        'w' to overwrite, 'a' to append.
    format : str
        Format string for the log messages.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_path = Path(filename)
    filename_path.parent.mkdir(parents=True, exist_ok=True)
    timestamped_filename = filename_path.with_name(
        f"{filename_path.stem}_{timestamp}{filename_path.suffix}"
    )
    resolved_filename = timestamped_filename.resolve()

    for handler in logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == resolved_filename
        ):
            return timestamped_filename

    file_handler = logging.FileHandler(
        timestamped_filename,
        mode=mode,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(file_handler)
    return timestamped_filename


@contextmanager
def use_log_level(verbose: Union[str, bool, int, None]):
    """Context manager for temporarily changing log level.
    
    Parameters
    ----------
    verbose : str, bool, int, or None
        The verbosity level to use temporarily
    
    Examples
    --------
    >>> with use_log_level('DEBUG'):
    ...     logger.debug('This will be printed')
    >>> logger.debug('This will not be printed')
    """
    old_level = set_log_level(verbose, return_old_level=True)
    try:
        yield
    finally:
        set_log_level(old_level)


@contextmanager
def use_console_log_level(verbose: Union[str, bool, int, None]):
    """Temporarily change console logging while keeping file logging intact.

    File handlers are deliberately excluded. This is useful when a long-running
    loop uses a terminal progress bar but the full INFO-level audit trail should
    still be written to the timestamped log file.
    """
    console_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    ]
    old_levels = [handler.level for handler in console_handlers]
    new_level = _parse_log_level(verbose)
    try:
        for handler in console_handlers:
            handler.setLevel(new_level)
        yield
    finally:
        for handler, old_level in zip(console_handlers, old_levels):
            handler.setLevel(old_level)


def verbose(function: Callable) -> Callable:
    """Decorator to allow functions to override log-level.
    
    Can be used on any function, regardless of whether it accepts a 'verbose' parameter.
    If the function does accept a 'verbose' parameter, that will be used.
    If not, the decorator will still work but the verbosity must be set when calling.
    
    Parameters
    ----------
    function : callable
        Function to be decorated
    
    Returns
    -------
    wrapper : callable
        Decorated function
    
    Examples
    --------
    # Function with verbose parameter
    >>> @verbose
    ... def func_with_verbose(x, verbose=None):
    ...     logger.info("Processing...")
    ...     return x
    
    # Function without verbose parameter
    >>> @verbose
    ... def func_without_verbose(x):
    ...     logger.info("Processing...")
    ...     return x
    
    # Both can be called with verbose parameter
    >>> result1 = func_with_verbose(10, verbose='DEBUG')
    >>> result2 = func_without_verbose(10, verbose='DEBUG')
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        # Check if 'verbose' is in kwargs
        verbose_level = kwargs.pop('verbose', None)
        
        # Check if the original function accepts 'verbose' parameter
        sig = inspect.signature(function)
        accepts_verbose = 'verbose' in sig.parameters
        
        # If function accepts verbose, pass it through
        if accepts_verbose:
            kwargs['verbose'] = verbose_level
            
        # Apply verbose level if specified
        if verbose_level is not None:
            with use_log_level(verbose_level):
                return function(*args, **kwargs)
        
        return function(*args, **kwargs)
    return wrapper

# Initialize logging when the module is imported
setup_logging()

# Example usage demonstrations
if __name__ == "__main__":
    # Example 1: Function with verbose parameter
    @verbose
    def process_with_verbose(x: Any, verbose: Optional[str] = None) -> None:
        logger.debug(f"Debug message: x = {x}")
        logger.info(f"Processing value: {x}")
        logger.warning("This is a warning")
    
    # Example 2: Function without verbose parameter
    @verbose
    def process_without_verbose(x: Any) -> None:
        logger.debug(f"Debug message: x = {x}")
        logger.info(f"Processing value: {x}")
        logger.warning("This is a warning")
    
    print("\n=== Testing function with verbose parameter ===")
    print("\nDefault level:")
    process_with_verbose(42)
    
    print("\nDebug level:")
    process_with_verbose(42, verbose='DEBUG')
    
    print("\nWarning level:")
    process_with_verbose(42, verbose='WARNING')
    
    print("\n=== Testing function without verbose parameter ===")
    print("\nDefault level:")
    process_without_verbose(42)
    
    print("\nDebug level:")
    process_without_verbose(42, verbose='DEBUG')
    
    print("\nWarning level:")
    process_without_verbose(42, verbose='WARNING')
    
    print("\n=== Testing context manager ===")
    with use_log_level('DEBUG'):
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        process_without_verbose(42)
    
    print("\n=== Testing file logging ===")
    set_log_file('example.log')
    logger.info("This message goes to both console and file")
