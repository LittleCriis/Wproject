import pandas as pd

df=pd.read_csv('C:/Users/Cristina/Desktop/MASTER DATA/Python/Wproject/data/Hombres.csv', sep=';', encoding='ISO-8859-1')


# Convertir todos los valores a numéricos, reemplazando comas por puntos
df = df.replace({',': '.'}, regex=True)

# Convertir las columnas que contienen números a tipo float
df = df.apply(pd.to_numeric, errors='coerce')

# Sumar los trimestres de cada año y agregar la columna con el resultado
df['2024'] = df['2024T1'] + df['2024T2'] + df['2024T3'] + df['2024T4']
df['2023'] = df['2023T1'] + df['2023T2'] + df['2023T3'] + df['2023T4']
df['2022'] = df['2022T1'] + df['2022T2'] + df['2022T3'] + df['2022T4']
# Repite para los otros años según sea necesario

# Eliminar las columnas de trimestres
df = df.drop(columns=[col for col in df.columns if 'T' in col])

# Guardar el archivo resultante
df.to_csv('C:/Users/Cristina/Desktop/MASTER DATA/Python/Wproject/data/Hombres_años.csv', index=False)