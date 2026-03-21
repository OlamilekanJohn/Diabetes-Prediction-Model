"""
Data preprocessing script. 
"""

import pandas as pd
def preprocess_data(df: pd.DataFrame, target_col: str='CLASS') -> pd.DataFrame:
    """Basic cleaning of data set.
    - handle missing values
    - feature engineering (correct data types, spacing issues with strings, and correct class labels in target column) etc.)
    """
    # Handle missing values drop rows with missing values 
    df.dropna(inplace=True)

    # drop ID and no_partion colums if they exist 
    for col in ['ID', 'No_Pation']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    # feature engineering: correct data types, spacing issues with strings, and correct class labels in target column
    for col in ['CLASS', 'Gender']: 
        if col in df.columns:
            df.loc[df[col] == 'P', col] = 'Y'
            df.loc[df[col] == 'f', col] = 'F'
            df.loc[df[col] == 'm', col] = 'M'
            df[col] = df[col].str.strip() 

    return df
