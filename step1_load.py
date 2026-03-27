import pandas as pd

# Load the dataset
df = pd.read_csv('merged_shark_data.csv')

# Print number of rows and column names
print(f"Number of rows: {len(df)}")
print(f"Column names: {list(df.columns)}")

# Print first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Basic summary stats for TL_cm and avg_pec_cm2
print("\nSummary stats for TL_cm:")
print(df['TL_cm'].describe())

print("\nSummary stats for avg_pec_cm2:")
print(df['avg_pec_cm2'].describe())

# Drop rows with missing values and print warnings
initial_rows = len(df)
df = df.dropna()
dropped_rows = initial_rows - len(df)
if dropped_rows > 0:
    print(f"\nWarning: Dropped {dropped_rows} rows with missing values.")
else:
    print("\nNo rows with missing values found.")

print(f"\nFinal number of rows: {len(df)}")
print("Data loaded cleanly.")