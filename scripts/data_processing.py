import pandas as pd

# Load the cleaned dataset
data = pd.read_csv('../data/All_data_daily_cleaned.csv')

# Convert each row to a text format
def row_to_text(row):
    # Convert each row into a descriptive sentence format
    return (
        f"On {row['Date']}, the mains power consumption was {row['mains']} units, "
        f"television used {row['television']} units, fan used {row['fan']} units, "
        f"fridge used {row['fridge']} units, and laptop computer used {row['laptop computer']} units. "
        f"Environmental conditions were: temperature {row['temp']}°C, dew point {row['dwpt']}°C, "
        f"relative humidity {row['rhum']}%, wind speed {row['wspd']} km/h, and pressure {row['pres']} hPa. "
        f"Day: {row['day_name']} ({row['month']}), Time of Day: {row['time_of_day']}, "
        f"Is Weekend: {row['is_weekend']}, Holiday: {row['Holiday']}."
    )

# Apply function to each row and create a new column for text representation
data['text'] = data.apply(row_to_text, axis=1)

# Save the text data for embedding
data[['text']].to_csv('../data/data_text_for_embedding.csv', index=False)

print("Data processing complete. Text data is ready for embedding.")
