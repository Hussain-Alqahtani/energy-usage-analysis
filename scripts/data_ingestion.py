import pandas as pd

# Load the dataset
data = pd.read_csv('../data/All_data_daily.csv')

# Drop unnecessary 'Unnamed: 0' column
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

# Display basic info
print("Dataset Info after cleaning:")
print(data.info())

# Check for missing values
print("\nMissing Values after cleaning:")
print(data.isnull().sum())

# Preview the first few rows
print("\nFirst few rows of the cleaned dataset:")
print(data.head())

# Save the cleaned data for future steps
data.to_csv('../data/All_data_daily_cleaned.csv', index=False)
