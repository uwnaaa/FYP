# If not already installed, do: pip install pandas fastparquet
import pandas as pd

URL_DATA = 'https://storage.data.gov.my/pricecatcher/lookup_premise.parquet'

location = pd.read_parquet(URL_DATA)
if 'date' in location.columns: location['date'] = pd.to_datetime(location['date'])

print(location)
df_ayam = df_ayam.merge(location, on = 'premise_code', how = 'left')
df_ayam
