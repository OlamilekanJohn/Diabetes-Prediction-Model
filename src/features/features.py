"""
Data preprocessing script. 
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder 

def encode_features(df: pd.DataFrame, target_col: str = 'CLASS') -> pd.DataFrame:
    """Apply one-hot encoding to categorical features.
    - Binary encoding for target column ('CLASS') """

    # Binary encoding for target column ('CLASS')
    for col in [target_col]:
        if col in df.columns:
            df.loc[df[col] == 'Y', col] = 1
            df.loc[df[col] == 'N', col] = 0
            df[col] = df[col].astype(int)
    
    # one-hot encoding for Gender categorical feature
    col = ['Gender']
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop first to avoid multicollinearity
    one_hot_encoded = one_hot_encoder.fit_transform(df[col])
    encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(col), index=df.index)
    encoded_df = pd.concat([df, encoded_df], axis=1).drop(col, axis=1)

    print(f"Feature engineering completed: {encoded_df.shape[1]} features after encoding.")
    return encoded_df
