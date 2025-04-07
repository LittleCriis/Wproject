import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los archivos CSV proporcionados
ambos_sexos = pd.read_csv('C:/Users/Cristina/Desktop/MASTER DATA/Python/Wproject/data/datosINEseparado.csv', sep=';', encoding='ISO-8859-1')
hombres = pd.read_csv('C:/Users/Cristina/Desktop/MASTER DATA/Python/Wproject/data/Hombres.csv', sep=';', encoding='ISO-8859-1')
mujeres = pd.read_csv('C:/Users/Cristina/Desktop/MASTER DATA/Python/Wproject/data/Mujeres.csv', sep=';', encoding='ISO-8859-1')

# Mostrar las primeras filas de cada dataset para entender su estructura
print(ambos_sexos.head())
print(hombres.head())
print(mujeres.head())

# Convertir las columnas de trimestres a tipo float para ambos_sexos, hombres y mujeres
for col in ambos_sexos.columns:
    if 'T' in col:  # Asegurarse de que solo estamos seleccionando las columnas de trimestres
        ambos_sexos[col] = pd.to_numeric(ambos_sexos[col].str.replace(',', '', regex=False), errors='coerce')

for col in hombres.columns:
    if 'T' in col:
        hombres[col] = pd.to_numeric(hombres[col].str.replace(',', '', regex=False), errors='coerce')

for col in mujeres.columns:
    if 'T' in col:
        mujeres[col] = pd.to_numeric(mujeres[col].str.replace(',', '', regex=False), errors='coerce')

# Eliminar la columna "Puesto" para análisis cuantitativos
ambos_sexos_cleaned = ambos_sexos.drop(columns=['Puesto'])
hombres_cleaned = hombres.drop(columns=['Puesto'])
mujeres_cleaned = mujeres.drop(columns=['Puesto'])

# Calcular las métricas para Hombres
hombres_cleaned['media'] = hombres_cleaned.mean(axis=1)
hombres_cleaned['mediana'] = hombres_cleaned.median(axis=1)
hombres_cleaned['moda'] = hombres_cleaned.mode(axis=1)[0]
hombres_cleaned['desviacion_estandar'] = hombres_cleaned.std(axis=1)

# Calcular las métricas para Mujeres
mujeres_cleaned['media'] = mujeres_cleaned.mean(axis=1)
mujeres_cleaned['mediana'] = mujeres_cleaned.median(axis=1)
mujeres_cleaned['moda'] = mujeres_cleaned.mode(axis=1)[0]
mujeres_cleaned['desviacion_estandar'] = mujeres_cleaned.std(axis=1)

# Verificar las primeras filas después de agregar las métricas
print(hombres_cleaned[['media', 'mediana', 'moda', 'desviacion_estandar']].head())
print(mujeres_cleaned[['media', 'mediana', 'moda', 'desviacion_estandar']].head())

# Asegurarnos de que las longitudes de Hombres y Mujeres coincidan
min_length = min(len(hombres_cleaned), len(mujeres_cleaned))

# Redimensionar ambos conjuntos de datos para tener la misma longitud
hombres_cleaned = hombres_cleaned.head(min_length)
mujeres_cleaned = mujeres_cleaned.head(min_length)

# Crear el DataFrame unificado para las métricas de Hombres y Mujeres
boxplot_data = pd.DataFrame({
    'Hombres': hombres_cleaned['media'].values,
    'Mujeres': mujeres_cleaned['media'].values
})

# Verificar la estructura de los datos antes de graficar
print(boxplot_data.head())

# Crear el gráfico de caja para la media comparando Hombres y Mujeres
plt.figure(figsize=(14, 6))

sns.boxplot(data=boxplot_data, palette="Set2")
plt.title('Distribución de la Media por Puesto (Hombres vs Mujeres)')
plt.xlabel('Género')
plt.ylabel('Valor')

plt.tight_layout()
plt.show()
