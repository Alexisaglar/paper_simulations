import pandas as pd
import glob
import matplotlib.pyplot as plt

# Pattern to match your CSV files
file_pattern = 'load_profiles/*.csv'
file_paths = glob.glob(file_pattern)

# Initialize an empty DataFrame for the combined data
combined_df = pd.DataFrame()

# Iterate over the file paths
for i, file_path in enumerate(file_paths):
    # Read the current CSV file
    temp_df = pd.read_csv(file_path)
    
    # Rename 'load consumption' to a unique name, e.g., using the file index or name
    column_name = f'profile_{i+1}'  # Or extract a meaningful name from file_path
    temp_df.rename(columns={'mult': column_name}, inplace=True)
    
    temp_df.set_index('time', inplace=True)

    # If it's the first file, initialize combined_df with this data
    if combined_df.empty:
        combined_df = temp_df
    else:
        combined_df = combined_df.join(temp_df, how='outer')
        print(i)

# Ensure the 'time' column is the DataFrame index (optional)
# combined_df.set_index('time', inplace=True)
combined_df.to_csv('file.csv')

# Identify the Maximum Value in Each Column
max_values = combined_df.max()

# Step 2: Normalize Each Column by Its Maximum Value
normalized_df = combined_df.divide(0.75, axis='columns')

# Step 3: Calculate the Average Profile
average_profile = normalized_df.mean(axis=1)

# Calculate the sum of energy consumption across all profiles for each timestamp
total_energy_consumption = combined_df.sum(axis=1)

# Plotting the Total Energy Consumption
plt.figure(figsize=(10, 6))
total_energy_consumption.plot(title='Total Energy Consumption Over Time')
plt.xlabel('Time')
plt.ylabel('Total Energy Consumption')
plt.grid(True)
plt.show()

# Plotting the Average Profile
plt.figure(figsize=(10, 6))
average_profile.plot(title='Average Profile Based on Maximum Values')
plt.xlabel('Time')
plt.ylabel('Normalized Consumption')
plt.grid(True)
plt.show()