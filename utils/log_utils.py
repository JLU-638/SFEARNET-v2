
import functools
import os
import sys
import logging
from typing import Union
from termcolor import colored


@functools.lru_cache()
def create_loggerGt(dirLog: Union[str, bytes, os.PathLike],
                    t: str,
                    dist_rank: int = 0,
                    name: str = "training"):
    """

    :param dirLog:
    :param t:
    :param dist_rank:
    :param name:
    :return:
    """
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + \
                ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers

    file_handler = logging.FileHandler(os.path.join(dirLog, f'{name}_{t}_rank-{dist_rank}.log'),
                                       mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger