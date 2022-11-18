"""
log utils
"""
import logging


def get_logger(name='root'):
    """
    get logger
    """
    formatter = logging.Formatter(
        # fmt='%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s')
        fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger_ = logging.getLogger(name)
    logger_.setLevel(logging.DEBUG)
    logger_.addHandler(handler)
    return logger_


logger = get_logger('root')
