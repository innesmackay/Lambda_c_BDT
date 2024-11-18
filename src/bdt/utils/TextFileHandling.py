from Log import *


def ReadList(filename, isFloat=False):
    """
    Read values from a file into a list.
    :param filename: name of file to be read.
    """
    with open(filename, "r") as f:
        initial_vals = f.readlines()
    vals = [
        float(v.replace("\n", "")) if isFloat else str(v.replace("\n", ""))
        for v in initial_vals
    ]
    return vals


def ParseCut(cut_string):
    """
    Parses cut read in from a settings file (which can't
    have spaces).
    :param cut_string: string of requirement to be parsed
    """
    cut = cut_string
    if "&" in cut:
        cut = cut.replace("&", " & ")
    if "|" in cut:
        cut = cut.replace("|", " | ")
    return cut


class Settings:
    """
    Class for reading inputs from a text file.
    In the text file the inputs are stored file line-
    by-line with the format:
        KEY : VALUE
    The inputs are stored in a dictionary to be
    accessed. If the line in the text file begins
    with a '*' character it will not be read (use-
    ful for comments etc.)
    """

    def __init__(self, filename, verbose=True):
        """
        * :param filename: name of file to read
        * :param verbose: verbosity boolean
        """
        self.verbose = verbose
        self.file = filename
        if self.verbose:
            info(f"Reading settings from {self.file}")
        self.dict = self.ReadSettings()

    def ReadSettings(self):
        """
        Read the settings from the file into
        a dictionary.
        """
        vals = {}
        with open(self.file, "r") as f:
            for line in f:
                words = line.split()
                if len(words) > 0:
                    if words[0] != "*":
                        vals[words[0]] = words[1]
        return vals

    def GetF(self, id):
        """
        Return the setting as a float.
        :param id: key in dictionary
        """
        return float(self.dict[id])

    def GetS(self, id):
        """
        Return the setting as a string.
        :param id: key in dictionary
        """
        return str(self.dict[id])

    def GetI(self, id):
        """
        Return the setting as an integer.
        :param id: key in dictionary
        """
        return int(self.dict[id])

    def GetKeys(self):
        """
        Return a list of the keys.
        """
        return list(self.dict.keys())
