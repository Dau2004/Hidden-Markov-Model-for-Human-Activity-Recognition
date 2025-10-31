import pandas as pd
import os

# Path to your Walking folder
walking_path = "/Users/ram/Development/HMM Data/Walking"

# Prepare a list to hold all dataframes
dataframes = []

# Loop through each subfolder (Walking_1, Walking_2, etc.)
for folder in os.listdir(walking_path):
    subfolder_path = os.path.join(walking_path, folder)
    if os.path.isdir(subfolder_path):
        acc_file = os.path.join(subfolder_path, "Accelerometer.csv")
        gyro_file = os.path.join(subfolder_path, "Gyroscope.csv")
        
        # Check if both files exist
        if os.path.exists(acc_file) and os.path.exists(gyro_file):
            # Read them
            acc_df = pd.read_csv(acc_file)
            gyro_df = pd.read_csv(gyro_file)
            
            # Add prefix to differentiate columns
            acc_df = acc_df.add_prefix('acc_')
            gyro_df = gyro_df.add_prefix('gyro_')
            
            # Try merging on timestamp (if both have one)
            if 'acc_timestamp' in acc_df.columns and 'gyro_timestamp' in gyro_df.columns:
                merged = pd.merge_asof(
                    acc_df.sort_values('acc_timestamp'),
                    gyro_df.sort_values('gyro_timestamp'),
                    left_on='acc_timestamp',
                    right_on='gyro_timestamp',
                    direction='nearest',
                    tolerance=0.02  # within 20ms, adjust if needed
                )
            else:
                merged = pd.concat([acc_df, gyro_df], axis=1)
            
            # Add metadata
            merged["activity"] = "Walking"
            merged["session"] = folder

            dataframes.append(merged)

# Concatenate all sessions into one big CSV
final_df = pd.concat(dataframes, ignore_index=True)

# Save to file
output_path = os.path.join(walking_path, "Walking_all.csv")
final_df.to_csv(output_path, index=False)

print(f"✅ Merged all Walking CSVs into: {output_path}")
print(f"Total rows: {len(final_df)}")
