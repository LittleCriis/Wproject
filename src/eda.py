import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los archivos CSV proporcionados
ambos_sexos = pd.read_csv('C:/Users/Cristina/Desktop/MASTER DATA/Python/Wproject/data/datosINEseparado.csv', sep=';', encoding='ISO-8859-1')
hombres = pd.read_csv('C:/Users/Cristina/Desktop/MASTER DATA/Python/Wproject/data/Hombres.csv', sep=';', encoding='ISO-8859-1')
mujeres = pd.read_csv('C:/Users/Cristina/Desktop/MASTER DATA/Python/Wproject/data/Mujeres.csv', sep=';', encoding='ISO-8859-1')

# Mostrar las primeras filas de cada dataset para entender su estructura
ambos_sexos.head(), hombres.head(), mujeres.head()

# Revisar las dimensiones de cada dataset
ambos_sexos_shape = ambos_sexos.shape
hombres_shape = hombres.shape
mujeres_shape = mujeres.shape

# Calcular estadísticas descriptivas para cada uno
ambos_sexos_desc = ambos_sexos.describe()
hombres_desc = hombres.describe()
mujeres_desc = mujeres.describe()

# Verificar si existen columnas no numéricas que debemos excluir para las estadísticas
ambos_sexos_non_numeric = ambos_sexos.select_dtypes(exclude=['number']).columns
hombres_non_numeric = hombres.select_dtypes(exclude=['number']).columns
mujeres_non_numeric = mujeres.select_dtypes(exclude=['number']).columns

# Mostrar la información relevante
print(ambos_sexos_shape, hombres_shape, mujeres_shape, ambos_sexos_desc.head(), hombres_desc.head(), mujeres_desc.head(), ambos_sexos_non_numeric, hombres_non_numeric, mujeres_non_numeric)

# Convertir las columnas de los trimestres a tipo float
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

# Asegurémonos de que no haya NaN en los trimestres
hombres_2024T4_cleaned = hombres_cleaned['2024T4'].dropna()
mujeres_2024T4_cleaned = mujeres_cleaned['2024T4'].dropna()

# Verificar las longitudes de los datos
print(f"Longitud de '2024T4' en Hombres después de dropna: {len(hombres_2024T4_cleaned)}")
print(f"Longitud de '2024T4' en Mujeres después de dropna: {len(mujeres_2024T4_cleaned)}")

# Crear el gráfico de caja con los datos de Hombres y Mujeres
plt.figure(figsize=(14, 6))

# Boxplot para los trimestres de 2024T4
sns.boxplot(data=[hombres_2024T4_cleaned, mujeres_2024T4_cleaned], labels=['Hombres', 'Mujeres'], palette="Set2")
plt.title('Distribución de Trimestres 2024T4 (Hombres vs Mujeres)')
plt.xlabel('Género')
plt.ylabel('Valor')

plt.tight_layout()
plt.show()