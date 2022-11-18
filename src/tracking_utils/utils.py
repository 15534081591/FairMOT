"""
track utils
"""
import os
import os.path as osp


def mkdir_if_missing(d):
    """mkdir if missing"""
    if not osp.exists(d):
        os.makedirs(d)
