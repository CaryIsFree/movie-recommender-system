import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#Load in the data using pandas
df = pd.read_csv('./data/movielens/movies_metadata.csv')
df_encoded = df.copy()
print(df_encoded)

#Encode all the binary features in the "movies_metadata.csv"
#TODO: Video
df_encoded['adult'] = df_encoded['adult'].map({'True': 1, 'False': 0})
print(df_encoded['adult'])

#Encode all the numerical features using MinMaxScaler
scaler = MinMaxScaler()

#vote_count
vote_count = df['vote_count']
print(f'{vote_count.max()=}')
print(f'{vote_count.min()=}')
print(f'{vote_count.mean()=}'),
print(f'{vote_count.median()=}'),
print(f'{vote_count.std()=}'),
print(vote_count.quantile([0.25, 0.5, 0.75, 0.95])) # Percentiles for the data
X_normalized = scaler.fit_transform(vote_count.to_numpy().reshape(-1, 1))
df_encoded['vote_count'] = X_normalized

#vote_average
vote_average = df['vote_average']
print(f'{vote_average.max()=}')
print(f'{vote_average.min()=}')
print(f'{vote_average.mean()=}'),
print(f'{vote_average.median()=}'),
print(f'{vote_average.std()=}'),
print(vote_average.quantile([0.25, 0.5, 0.75, 0.95])) # Percentiles for the data
X_normalized = scaler.fit_transform(vote_average.to_numpy().reshape(-1, 1))
df_encoded['vote_average'] = X_normalized

#revenue
revenue = df['revenue']
print(f'{revenue.max()=}')
print(f'{revenue.min()=}')
print(f'{revenue.mean()=}'),
print(f'{revenue.median()=}'),
print(f'{revenue.std()=}'),
print(revenue.quantile([0.25, 0.5, 0.75, 0.95])) # Percentiles for the data
X_normalized = scaler.fit_transform(revenue.to_numpy().reshape(-1, 1))
df_encoded['revenue'] = X_normalized

#runtime
runtime = df['runtime']
print(f'{runtime.max()=}')
print(f'{runtime.min()=}')
print(f'{runtime.mean()=}'),
print(f'{runtime.median()=}'),
print(f'{runtime.std()=}'),
print(runtime.quantile([0.25, 0.5, 0.75, 0.95])) # Percentiles for the data
X_normalized = scaler.fit_transform(runtime.to_numpy().reshape(-1, 1))
df_encoded['runtime'] = X_normalized

#popularity
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce') #Converts popularity column as a numeric (originally a string)

popularity = df['popularity']
print(f'{popularity.max()=}')
print(f'{popularity.min()=}')
print(f'{popularity.mean()=}'),
print(f'{popularity.median()=}'),
print(f'{popularity.std()=}'),
print(popularity.quantile([0.25, 0.5, 0.75, 0.95])) # Percentiles for the data
X_normalized = scaler.fit_transform(popularity.to_numpy().reshape(-1, 1))
df_encoded['popularity'] = X_normalized