from pandas import read_csv, to_datetime, to_numeric
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from math import sqrt

np.random.seed(0)
data = read_csv("Documents/Scolaire/2018-2019 ISIMA/flights.csv")
#data = read_csv("Documents/Scolaire/2018-2019 ISIMA/Projet/test.csv")
n=np.shape(data)[0]
data["CANCELLED"].value_counts()
data["DIVERTED"].value_counts()
#print(data.loc[:,'DIVERTED'].describe())

## DATA CLEANING
#Suppresion des attributs jugés peu pertinents
list=['TAXI_IN','TAXI_OUT','WHEELS_OFF','WHEELS_ON','ELAPSED_TIME','AIR_TIME','CANCELLATION_REASON','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY']
data_cleaned = data.copy(deep=True)
data_cleaned = data_cleaned.drop(labels=list,axis=1)

'''
missing_values = data.isnull().sum()
print(100*missing_values/(n)) #Pourcentage de données manquantes par attribut
'''

#Suppression des lignes où la donnée sur le retard d'arrivée est manquante
data_cleaned = data_cleaned.dropna(axis=0,subset=["ARRIVAL_DELAY"]) 

#Mise au format de la date
data_cleaned['DATE'] = to_datetime(data_cleaned[['YEAR','MONTH', 'DAY']])
data_cleaned = data_cleaned.drop(labels=['YEAR','MONTH','DAY'],axis=1)


n_cleaned=np.shape(data_cleaned)[0] #Nombre de lignes restantes
print(100*data_cleaned.isnull().sum()/n_cleaned) #Pourcentage de données manquantes par attribut

#Conversion des données entières vers float
data_cleaned.loc['SCHEDULED_TIME', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'DISTANCE', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY'].values.astype(float)

#print(data_cleaned.loc[:,'DIVERTED'].describe())
#print(data_cleaned.tail(3))
#print(data_cleaned.head(5))

## ETUDES DES CORRELATIONS
Attribut_principal = ['DESTINATION_AIRPORT']
Attribut_etude = ['SCHEDULED_TIME', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'SCHEDULED_TIME', 'DISTANCE', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY']
p=len(Attribut_etude)

#Normalisation pour l'ACP
X = data_cleaned.loc[:,Attribut_etude]
sc = StandardScaler()
Z = sc.fit_transform(X)

#ACP
acp = PCA(svd_solver='full', n_components=None)
acp.fit_transform(Z)

eigval =(p-1)*acp.explained_variance_/p
print(eigval)

corvar = np.zeros((p,p))
for k in range(p):
    corvar[:,k] = acp.components_[k,:] * sqrt(eigval[k])

print(corvar)

##