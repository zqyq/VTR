import torch
import logging


def debug(str):
    if torch.distributed.get_rank() == 0:
        logging.debug(str)


def info(str):
    if torch.distributed.get_rank() == 0:
        logging.info(str)


def warn(str):
    if torch.distributed.get_rank() == 0:
        logging.warn(str)

def error(str):
    if torch.distributed.get_rank() == 0:
        logging.error(str)
