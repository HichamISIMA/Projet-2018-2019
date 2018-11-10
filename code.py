import pandas as pd
import numpy as np


# read in data
airports_data = pd.read_csv("Documents/Scolaire/2018-2019 ISIMA/Projet/airports.csv")
#flights_data = pd.read_csv("Documents/Scolaire/2018-2019 ISIMA/Projet/flights.csv")

# set seed faor reproducibility
np.random.seed(0)

print(airports_data.head(5))

print(airports_data.sample(6))

#flights_data.np.sample(5)

