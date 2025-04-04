import pandas as pd 

#Leemos el csv con el que vamos a trabajar y lo guardamos en un dataframe
df = pd.read_csv('C:/Users/Cristina/Desktop/MASTER DATA/Python/Wproject/data/datosINE.csv', sep=';', encoding='ISO-8859-1')
#Mostramos el dataframe
print(df.head(10))
#Mostramos la información del dataframe
df.info()
#Mostramos la descripción del dataframe
df.describe()
#Mostramos la cantidad de filas y columnas del dataframe
df.shape
#Mostramos los tipos de datos del dataframe
df.dtypes
#Eliminamos todas las filas hasta la número 6
df = df.iloc[6:]
#Sustituir las celdas NaN por un string vacío
df = df.fillna('')
# Renombrar columnas "Unnamed: X" con un string vacío
df.columns = [col if not col.startswith('Unnamed:') else '' for col in df.columns]
# Colocar "Años" solo en la celda A9 (fila 8, columna 0)
df.at[7, df.columns[0]] = 'Años'
#Mostramos el dataframe 
print(df.head(10))
