import os
from Log import info


def CheckDir(path):
    """
    Check if a directory already exists.
    If not then make it.
    :param path: path to check
    """
    if os.path.exists(path):
        info("{} directory already exists".format(path))
    else:
        info("Making directory {}".format(path))
        os.makedirs(path)
    return
