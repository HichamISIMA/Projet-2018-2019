import pandas as pd
import numpy as np

#flights_data = pd.read_csv("Documents/Scolaire/2018-2019 ISIMA/Projet/flights.csv")
#flights_data.sample(5)

data = pd.read_csv("Documents/Scolaire/2018-2019 ISIMA/Projet/airports.csv")

#print(data.sample(5))

s = pd.DataFrame(np.random.randn(50))
s.head()
print(s.sample(n=3))

missing_values = data.isnull().sum()

#print(missing_values)

data.dropna()
print(data.head())
columns_with_na_dropped = data.dropna(axis=1)
print(columns_with_na_dropped.head())