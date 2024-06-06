# If not already installed, do: pip install pandas fastparquet
import pandas as pd

URL_DATA = 'https://storage.dosm.gov.my/hies/hies_district.parquet'

df = pd.read_parquet(URL_DATA)
if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'])

# Define the list of districts in Selangor
selangor_districts = [
    'Gombak', 'Ulu Langat', 'Ulu Selangor', 'Klang', 'Kuala Langat',
    'Kuala Selangor', 'Petaling', 'Petaling Jaya', 'Rawang' 'Sabak Bernam', 'Sepang'
]

# Filter the DataFrame for Selangor districts only
selangor_df = df[df['district'].isin(selangor_districts)].copy()

# Remove the 'state' column
selangor_df.drop(columns=['state'], inplace=True)

# Display the DataFrame to confirm the state column has been removed and only Selangor districts are included
print(selangor_df)

