import pandas as pd
import os

def load_data(filepath):
    """
    Loads the advertising dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} was not found.")
    
    # The first column is an index column
    df = pd.read_csv(filepath, index_col=0)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df
