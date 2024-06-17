import os
import numpy as np

def calculate_batch_statistics(files, directory, size_info):
    batch_data = []
    for file in files:
        npy_path = os.path.join(directory, file)
        data = np.load(npy_path)
        batch_data.append(data)
        # Update file size info for each .npy file
        file_size_gb = os.path.getsize(npy_path) / (1024 ** 3)  # Size in GB
        size_info['processed_files_size'] += file_size_gb
        size_info['files_5GB_counter'] += file_size_gb
        size_info['processed_files_count'] += 1

    # Combine all data into one array
    if batch_data:
        print(f"Processed {size_info['processed_files_size']:.2f} GB of audio files ({size_info['processed_files_count']} files)")
        batch_combined_data = np.concatenate(batch_data, axis=None)  # Flatten and combine data
        batch_std = np.std(batch_combined_data)
        batch_mean = np.mean(batch_combined_data)
        size_info['total_mel_size'] += batch_combined_data.nbytes / (1024 ** 3)  # Update total size of all arrays in GB
        return batch_std, batch_mean
    else:
        return None, None

def process_directories(base_directory):
    size_info = {
        'total_mel_size': 0,
        'processed_files_size': 0,
        'files_5GB_counter': 0,
        'processed_files_count': 0
    }
    batch_results = []
    subdir_count = 0

    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)
        if os.path.isdir(subdir_path):
            all_files = [f for f in os.listdir(subdir_path) if f.endswith('.npy')]
            batch_size = 2048
            for i in range(0, len(all_files), batch_size):
                batch_files = all_files[i:i + batch_size]
                batch_std, batch_mean = calculate_batch_statistics(batch_files, subdir_path, size_info)
                if batch_std is not None and batch_mean is not None:
                    batch_results.append((subdir, batch_std, batch_mean))
            subdir_count += 1  # Increment subdirectory counter
            print(f"Total data size of saved arrays: {size_info['total_mel_size']:.2f} GB")
            print(f"Total subdirectories processed: {subdir_count}")

    return batch_results

def save_to_csv(batch_results, output_file):
    import csv
    # Prepare the structure to aggregate batch_results
    aggregated_batch_results = {}
    for subdir, std, mean in batch_results:
        if subdir not in aggregated_batch_results:
            aggregated_batch_results[subdir] = {'stds': [], 'means': []}
        aggregated_batch_results[subdir]['stds'].append(std)
        aggregated_batch_results[subdir]['means'].append(mean)

    # Write batch_results to a CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'std', 'mean'])
        # Calculate the average of std and mean for each subdirectory and write to the file
        for subdir, stats in aggregated_batch_results.items():
            avg_std = np.mean(stats['stds'])
            avg_mean = np.mean(stats['means'])
            writer.writerow([subdir, avg_std, avg_mean])

    print(f"batch_Results have been saved in {output_file}.")

base_directory = 'data/spectrograms'
batch_results = process_directories(base_directory)
output_csv_path = 'data/combined_spectrogram_statistics.csv'
save_to_csv(batch_results, output_csv_path)

print(f'batch_Results have been saved in {output_csv_path}.')


