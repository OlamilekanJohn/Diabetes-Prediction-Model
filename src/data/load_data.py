"""
Basic data processing script template.
"""

import pandas as pd
import os


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data into a DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        pd.DataFrame containing the loaded dataset
    """
    
    return pd.read_csv(file_path)


