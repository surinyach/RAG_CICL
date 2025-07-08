import pandas as pd
import pickle

def load_data(path):
    """
    Loads the data contained in a pickle input file.

    Args:
        path(str): Location where the pickle (.pkl) file is located within the file system.
    
    Returns:
        data: A pandas dataframe with the information contained in the input file.
    """
    