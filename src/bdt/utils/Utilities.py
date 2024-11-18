import os
from logzero import logger as log


def CheckDir(path):
    """Check if a directory already exists. If not then make it.

    :param path: path to check.
    """
    if os.path.exists(path):
        log.info("{} directory already exists".format(path))
    else:
        log.info("Making directory {}".format(path))
        os.makedirs(path)
    return


def ParseCut(cut_string):
    """
    Parses cut read in from a settings file (which can't
    have spaces).

    :param cut_string: string of requirement to be parsed.
    :return: parsed string.
    """
    cut = cut_string
    if "&" in cut:
        cut = cut.replace("&", " & ")
    if "|" in cut:
        cut = cut.replace("|", " | ")
    return cut
