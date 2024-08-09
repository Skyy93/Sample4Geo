from config import cfg
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load the metadata CSV file
meta_df = pd.read_csv(os.path.join(cfg.data_path, "final_metadata.csv"), index_col=False)

# Remove the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in meta_df.columns:
    meta_df.drop('Unnamed: 0', axis=1, inplace=True)

# Calculate the total number of samples to select (10% of the total data)
total_sample_size = int(0.1 * len(meta_df))

# Calculate the number of samples per continent for a uniform distribution
samples_per_continent = int(total_sample_size / meta_df['continent'].nunique())

# Collecting uniform samples across continents
uniform_samples = []
remaining_sample_size = total_sample_size

for continent in meta_df['continent'].unique():
    continent_df = meta_df[meta_df['continent'] == continent]
    if len(continent_df) >= samples_per_continent:
        sampled_df = continent_df.sample(n=samples_per_continent, random_state=42)
        remaining_sample_size -= samples_per_continent
    else:
        sampled_df = continent_df  # take all available if less than needed
        remaining_sample_size -= len(sampled_df)
    uniform_samples.append(sampled_df)

# If any remainder from rounding, distribute these among the larger groups
if remaining_sample_size > 0:
    extra_samples = meta_df[~meta_df.index.isin(pd.concat(uniform_samples).index)]
    extra_sampled_df = extra_samples.sample(n=remaining_sample_size, random_state=42)
    uniform_samples.append(extra_sampled_df)

reduced_meta_df = pd.concat(uniform_samples)

# Print the number of samples per continent in the reduced dataset
for continent in reduced_meta_df['continent'].unique():
    count = reduced_meta_df[reduced_meta_df['continent'] == continent].shape[0]
    print(f"Number of samples for {continent}: {count}")

# Shuffle and split the reduced DataFrame into training, validation, and test sets
train_df, validate_df, test_df = np.split(
    reduced_meta_df.sample(frac=1, random_state=42), 
    [int(.70 * len(reduced_meta_df)), int(.8 * len(reduced_meta_df))]
)

# Print the sample counts for each dataset
print("Sample count - train/val/test for ratio 70:10:20:", (len(train_df), len(validate_df), len(test_df)))

# Define path where the splits will be saved
splits_path = cfg.data_path

# Save the dataframes to CSV without the index
train_df.to_csv(os.path.join(splits_path, 'small_train_df.csv'), index=False)
validate_df.to_csv(os.path.join(splits_path, 'small_validate_df.csv'), index=False)
test_df.to_csv(os.path.join(splits_path, 'small_test_df.csv'), index=False)