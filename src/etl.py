import pandas as pd 

# Leemos el csv con el que vamos a trabajar y lo guardamos en un dataframe
df = pd.read_csv('C:/Users/Cristina/Desktop/MASTER DATA/Python/Wproject/data/datosINE.csv', sep=';', encoding='ISO-8859-1')

# Mostramos el dataframe
print(df.head(10))

# Mostramos la información del dataframe
df.info()

# Mostramos la descripción del dataframe
df.describe()

# Mostramos la cantidad de filas y columnas del dataframe
print(f'Cantidad de filas y columnas: {df.shape}')

# Mostramos los tipos de datos del dataframe
print(f'Tipos de datos: {df.dtypes}')

# Eliminamos todas las filas hasta la número 6
df = df.iloc[5:]

# Sustituir las celdas NaN por un string vacío
df = df.fillna('')

# Renombrar columnas "Unnamed: X" con un string vacío
df.columns = [col if not col.startswith('Unnamed:') else '' for col in df.columns]

# Convertir las columnas de trimestres a valores numéricos, eliminando comas si es necesario
for col in df.columns:
    # Comprobamos que la columna corresponda a un trimestre
    if any(year in col for year in ['2024', '2023', '2022', '2021', '2020', '2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011']):
        # Limpiar los valores eliminando comas y convertir a numérico
        df[col] = pd.to_numeric(df[col].str.replace(',', '', regex=False), errors='coerce')

# Seleccionar las columnas correspondientes a los trimestres para cada año
year_columns = [col for col in df.columns if '2024' in col or '2023' in col or '2022' in col or '2021' in col or '2020' in col or '2019' in col or '2018' in col or '2017' in col or '2016' in col or '2015' in col or '2014' in col or '2013' in col or '2012' in col or '2011' in col]

# Calcular la suma de los trimestres para cada año y agregarlo como nuevas columnas
for year in range(2024, 2010, -1):  # Desde 2024 hasta 2011
    year_str = str(year)
    # Seleccionar las columnas correspondientes a ese año
    year_trimesters = [col for col in year_columns if year_str in col]
    # Calcular la suma de los trimestres y agregarlo como una nueva columna al final
    df[f'{year_str} Total'] = df[year_trimesters].sum(axis=1)

# Colocar "Años" solo en la celda A9 (fila 8, columna 0)
df.at[7, df.columns[0]] = 'Años'

# Mostramos el dataframe 
print(df.head(10))

# Guardar el dataframe actualizado en un archivo CSV (opcional)
df.to_csv('C:/Users/Cristina/Desktop/MASTER DATA/Python/Wproject/data/datosINE_agrupado.csv', sep=';', encoding='ISO-8859-1', index=False)
