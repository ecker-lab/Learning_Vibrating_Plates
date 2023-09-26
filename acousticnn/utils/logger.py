import logging
import os
import shutil
import time
from torch import distributed as dist


def is_main_process():
    rank, _ = get_dist_info()
    return rank == 0


# RETURNS UNIQUE IDENTIFIER OF THE CURRENT PROCESS WITHIN DISTRIBUTED PROCESS GROUP (rank) AND THE NUMBER OF SUCH PROCESSES - 1 (world_size)
def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger('acoustics')
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    if not is_main_process():
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


def init_train_logger(args, config):
    # Make sure previous handlers are removed
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    save_directory = args.dir
    os.makedirs(os.path.abspath(save_directory), exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(save_directory, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)
    logger.info(f'Config:\n{args}')
    logger.info(f'Config:\n{config}')
    shutil.copy(args.config, os.path.join(save_directory, os.path.basename(args.config)))
    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.
    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, '
            f'"silent" or None, but got {type(logger)}')


def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
