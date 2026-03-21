import os, sys
import pandas as pd

# make src directory available for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import preprocess_data
from src.features.features import encode_features

RAW= '/Users/macbook/Desktop/Python files/First deployed Project/data/raw/diabetes_unclean.csv' 
PROCESSED = '/Users/macbook/Desktop/Python files/First deployed Project/data/processed/diabetes_processed.csv'

# Load raw data

df = pd.read_csv(RAW)

# Preprocess data

df = preprocess_data(df, target_col='CLASS')

# ensure target is 0/1 only if still object
if 'CLASS' in df.columns and df['CLASS'].dtype == 'object':
    df['CLASS'] = df['CLASS'].str.strip().map({'Y': 1, 'N': 0}).astype(int)

# sanity check to ensure no missing values after preprocessing
assert df['CLASS'].isnull().sum() == 0, "Missing values found in target column after preprocessing"
assert set(df['CLASS'].unique()) <= {0, 1}, "class labels not 0/1 after preprocessing"

# Build features
df_processed = encode_features(df, target_col='CLASS')

# Save processed data
os.makedirs(os.path.dirname(PROCESSED), exist_ok=True)
df_processed.to_csv(PROCESSED, index=False)
print(f"Processed dataset saved to {PROCESSED} | Shape: {df_processed.shape}") 
    