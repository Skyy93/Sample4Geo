from config import cfg
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Loading the metadata CSV
meta_df = pd.read_csv(os.path.join(cfg.data_path, "final_metadata.csv"), index_col=False)

# Check and remove the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in meta_df.columns:
    meta_df.drop('Unnamed: 0', axis=1, inplace=True)

# Randomly shuffling and splitting the dataframe into train, validate, and test sets
train_df, validate_df, test_df = np.split(
    meta_df.sample(frac=1, random_state=42), 
    [int(.70*len(meta_df)), int(.8*len(meta_df))]
)

# Printing out the sample counts for each dataset
print("Sample count - train/val/test for ratio 70:10:20", (len(train_df), len(validate_df), len(test_df)))  # Example output: (35554, 5079, 10159)

# Define the path where the splits will be saved
splits_path = cfg.data_path

# Save the dataframes to CSV without the index
train_df.to_csv(os.path.join(splits_path, 'train_df.csv'), index=False)
validate_df.to_csv(os.path.join(splits_path, 'validate_df.csv'), index=False)
test_df.to_csv(os.path.join(splits_path, 'test_df.csv'), index=False)