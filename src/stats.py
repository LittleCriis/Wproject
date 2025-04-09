import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Cargar los archivos CSV proporcionados
ambos_sexos = pd.read_csv('C:/Users/Cristina/Desktop/MASTER DATA/Python/Wproject/data/datosINEseparado.csv', sep=';', encoding='ISO-8859-1')
hombres = pd.read_csv('C:/Users/Cristina/Desktop/MASTER DATA/Python/Wproject/data/Hombres.csv', sep=';', encoding='ISO-8859-1')
mujeres = pd.read_csv('C:/Users/Cristina/Desktop/MASTER DATA/Python/Wproject/data/Mujeres.csv', sep=';', encoding='ISO-8859-1')

# Convertir las columnas de trimestres a tipo float
def clean_and_convert(df):
    for col in df.columns:
        if 'T' in col:  # Asegurarse de que solo estamos seleccionando las columnas de trimestres
            # Reemplazar las comas por puntos y convertir a float
            df[col] = pd.to_numeric(df[col].str.replace(',', '.', regex=False), errors='coerce')
    return df

# Limpiar y convertir los valores
ambos_sexos = clean_and_convert(ambos_sexos)
hombres = clean_and_convert(hombres)
mujeres = clean_and_convert(mujeres)

# Eliminar la columna "Puesto" para análisis cuantitativos
ambos_sexos_cleaned = ambos_sexos.drop(columns=['Puesto'])
hombres_cleaned = hombres.drop(columns=['Puesto'])
mujeres_cleaned = mujeres.drop(columns=['Puesto'])

# Asegurarse de que el índice sea correcto
hombres_cleaned.index = hombres['Puesto']
mujeres_cleaned.index = mujeres['Puesto']

# Imputación de valores faltantes con la media
imputer = SimpleImputer(strategy='mean')
hombres_cleaned_imputed = pd.DataFrame(imputer.fit_transform(hombres_cleaned), columns=hombres_cleaned.columns, index=hombres_cleaned.index)
mujeres_cleaned_imputed = pd.DataFrame(imputer.fit_transform(mujeres_cleaned), columns=mujeres_cleaned.columns, index=mujeres_cleaned.index)

# Crear la columna 'media' en ambos DataFrames
hombres_cleaned_imputed['media'] = hombres_cleaned_imputed.mean(axis=1)
mujeres_cleaned_imputed['media'] = mujeres_cleaned_imputed.mean(axis=1)

# Recalcular las estadísticas descriptivas para Hombres y Mujeres por Puesto
hombres_cleaned_stats = hombres_cleaned_imputed.T.describe().T[['mean', '50%', 'std', 'min', 'max', '25%', '75%']]
mujeres_cleaned_stats = mujeres_cleaned_imputed.T.describe().T[['mean', '50%', 'std', 'min', 'max', '25%', '75%']]

# Calcular la moda por puesto
hombres_cleaned_stats['moda'] = hombres_cleaned_imputed.mode(axis=0).iloc[0]
mujeres_cleaned_stats['moda'] = mujeres_cleaned_imputed.mode(axis=0).iloc[0]

# Mostrar estadísticas descriptivas por puesto
print("Estadísticas Descriptivas para Hombres por Puesto:\n", hombres_cleaned_stats.head())
print("\nEstadísticas Descriptivas para Mujeres por Puesto:\n", mujeres_cleaned_stats.head())

# Crear DataFrame para los boxplots
df_boxplot = pd.DataFrame({
    'Hombres': hombres_cleaned_imputed['media'],
    'Mujeres': mujeres_cleaned_imputed['media']
})

# Mostrar boxplot para Hombres y Mujeres
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_boxplot)
plt.title('Distribución de la Media por Puesto (Hombres vs Mujeres)')
plt.xlabel('Género')
plt.ylabel('Valor')
plt.tight_layout()
plt.show()

# Estadísticas de dispersión: Histograma y KDE
plt.figure(figsize=(14, 6))

# Histograma y KDE
plt.subplot(1, 2, 1)
sns.histplot(hombres_cleaned_imputed['media'], kde=True, color="blue", label="Hombres")
sns.histplot(mujeres_cleaned_imputed['media'], kde=True, color="red", label="Mujeres")
plt.title('Histograma de la Media por Puesto')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()

# Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(data=df_boxplot)
plt.title('Distribución de la Media por Puesto (Hombres vs Mujeres)')
plt.xlabel('Género')
plt.ylabel('Valor')

plt.tight_layout()
plt.show()

# Estadística Inferencial
# Correlación entre variables numéricas
correlacion = hombres_cleaned_imputed.corrwith(mujeres_cleaned_imputed)
print("Correlaciones entre Hombres y Mujeres por Puesto:\n", correlacion)

# Contraste de hipótesis: Comparación de medias usando prueba t para dos muestras
t_stat, p_value = stats.ttest_ind(mujeres_cleaned_imputed['media'].dropna(), hombres_cleaned_imputed['media'].dropna())
print(f"\nResultado de la prueba t (Hombres vs Mujeres): T-statistic = {t_stat}, P-value = {p_value}")

# Si el P-value es menor que 0.05, podemos rechazar la hipótesis nula de que las medias son iguales

# Modelo de regresión lineal: Plantear un modelo de regresión lineal sobre una variable dependiente
# En este caso, vamos a realizar una regresión lineal para predecir la 'media' de mujeres en función de las demás variables
X = mujeres_cleaned_imputed.drop(columns=['media'])
y = mujeres_cleaned_imputed['media']

# Ajustar el modelo de regresión lineal
regressor = LinearRegression()
try:
    regressor.fit(X, y)
    print("\nResultados de la regresión lineal:")
    print(f"Coeficientes: {regressor.coef_}")
    print(f"Intercepto: {regressor.intercept_}")
except ValueError as e:
    print(f"No hay suficientes datos para la regresión: {e}")

# Predicción para los datos de entrada
try:
    y_pred = regressor.predict(X)
    print(f"Predicciones para los primeros 5 valores: {y_pred[:5]}")
except:
    print("No fue posible hacer la predicción debido a la falta de datos.")
