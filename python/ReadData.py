import numpy as np
def ReadData(infile, cols):
    """
    Reads in data file with predetermined number of columns
    Inputs:
    infile: file name string
    cols: number of columns the data is assumed to have.
    """
    data = np.loadtxt(infile)
    if data.shape[1] != cols:
        raise ValueError(f"Data file '{infile}' is assumed to have {cols} columns")
    return data