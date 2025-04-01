import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Leemos el csv con el que vamos a trabajar y lo guardamos en un dataframe
df=pd.read_csv('C:/Users/Cristina/Desktop/MASTER DATA/Python/Wproject/data/datosINE.csv', sep=';', encoding='ISO-8859-1', decimal=',')
#Mostramos el dataframe
print(df.head(10))
#Mostramos la información del dataframe
df.info()
#Mostramos la descripción del dataframe
df.describe()
#Mostramos la cantidad de filas y columnas del dataframe
df.shape
#Mostramos los nombres de las columnas del dataframe
df.columns
#Mostramos los tipos de datos del dataframe
df.dtypes