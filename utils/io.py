import time
import os
import polars as pl
import pyarrow.parquet as pq


def read_data(filepath, filetype):
    """
    Read data from file based on file type.
    Parameters:
        filepath (str): Path to the file.
        filetype (str): Type of the file ('csv' or 'parquet').
    Returns:
        DataFrame: DataFrame containing the data.
    """
    start_time = time.time()  # Record the start time
    file_name = os.path.basename(filepath) 
    if filetype == 'csv':
        df = pl.read_csv(filepath)
    elif filetype == 'parquet':
        df = pl.read_parquet(filepath)
    else:
        raise ValueError("Unsupported file type. Please provide either 'csv' or 'parquet'.")

    end_time = time.time()  # Record the end time
    load_time = end_time - start_time  # Calculate the loading time

    # Calculate the size of the loaded dataset in MB
    dataset_size_mb = df.estimated_size() / (1024 * 1024)
    print(f"File name: {file_name}")
    print(f"Time taken to load the dataset: {load_time:.2f} seconds")
    print(f"Size of the loaded dataset: {dataset_size_mb:.2f} MB\n")
    
    return df