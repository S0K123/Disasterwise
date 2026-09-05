import logging
import os

def setup_logger(log_file='experiments/disasterwise.log', level='INFO'):
    """
    Sets up a logger with both console and file handlers.
    
    Args:
        log_file (str): Path to the log file.
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        
    Returns:
        logging.Logger: Configured logger object.
    """
    # Create the directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    # Create logger
    logger = logging.getLogger('disasterwise')
    logger.setLevel(numeric_level)

    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
