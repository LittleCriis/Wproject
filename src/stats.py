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

# Imputar los NaN para evitar problemas con los modelos y gráficos
imputer = SimpleImputer(strategy='mean')  # Usamos la media para imputar los valores faltantes
hombres_cleaned = pd.DataFrame(imputer.fit_transform(hombres_cleaned), columns=hombres_cleaned.columns, index=hombres_cleaned.index)
mujeres_cleaned = pd.DataFrame(imputer.fit_transform(mujeres_cleaned), columns=mujeres_cleaned.columns, index=mujeres_cleaned.index)

# Verificar que hay suficientes datos para la regresión
if hombres_cleaned.shape[0] > 0 and mujeres_cleaned.shape[0] > 0:
    # Calcular las métricas para Hombres y Mujeres por Puesto
    hombres_cleaned_stats = hombres_cleaned.T.describe().T[['mean', '50%', 'std', 'min', 'max', '25%', '75%']]
    mujeres_cleaned_stats = mujeres_cleaned.T.describe().T[['mean', '50%', 'std', 'min', 'max', '25%', '75%']]

    # Calcular la moda por puesto
    hombres_cleaned_stats['moda'] = hombres_cleaned.mode(axis=0).iloc[0]
    mujeres_cleaned_stats['moda'] = mujeres_cleaned.mode(axis=0).iloc[0]

    # Mostrar estadísticas descriptivas por puesto
    print("Estadísticas Descriptivas para Hombres por Puesto:\n", hombres_cleaned_stats.head())
    print("\nEstadísticas Descriptivas para Mujeres por Puesto:\n", mujeres_cleaned_stats.head())

    # Volver a calcular las métricas de 'media', 'mediana', 'moda', etc., asegurándonos de que las columnas estén presentes
    hombres_cleaned['media'] = hombres_cleaned.mean(axis=1)
    mujeres_cleaned['media'] = mujeres_cleaned.mean(axis=1)
    hombres_cleaned['mediana'] = hombres_cleaned.median(axis=1)
    mujeres_cleaned['mediana'] = mujeres_cleaned.median(axis=1)
    hombres_cleaned['moda'] = hombres_cleaned.mode(axis=1).iloc[0]
    mujeres_cleaned['moda'] = mujeres_cleaned.mode(axis=1).iloc[0]
    hombres_cleaned['desviacion_estandar'] = hombres_cleaned.std(axis=1)
    mujeres_cleaned['desviacion_estandar'] = mujeres_cleaned.std(axis=1)

    # Estadísticas por subgrupos (segmentación por género) 
    # Comparación de la media de hombres vs mujeres
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=[hombres_cleaned['media'], mujeres_cleaned['media']])
    plt.title('Distribución de la Media por Puesto (Hombres vs Mujeres)')
    plt.xlabel('Género')
    plt.ylabel('Valor')
    plt.xticks([0, 1], ['Hombres', 'Mujeres'])  # Asegurarse de etiquetar correctamente
    plt.tight_layout()
    plt.show()

    # Estadísticas de dispersión: Histograma y KDE
    plt.figure(figsize=(14, 6))

    # Histograma y KDE
    plt.subplot(1, 2, 1)
    sns.histplot(hombres_cleaned['media'], kde=True, color="blue", label="Hombres")
    sns.histplot(mujeres_cleaned['media'], kde=True, color="red", label="Mujeres")
    plt.title('Histograma de la Media por Puesto')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.legend()

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(data=[hombres_cleaned['media'], mujeres_cleaned['media']])
    plt.title('Distribución de la Media por Puesto (Hombres vs Mujeres)')
    plt.xlabel('Género')
    plt.ylabel('Valor')
    plt.xticks([0, 1], ['Hombres', 'Mujeres']) 

    plt.tight_layout()
    plt.show()

    # Estadística Inferencial
    # Correlación entre variables numéricas
    correlacion = hombres_cleaned.corrwith(mujeres_cleaned)
    print("Correlaciones entre Hombres y Mujeres:\n", correlacion)

    # Contraste de hipótesis: Comparación de medias usando prueba t para dos muestras
    t_stat, p_value = stats.ttest_ind(mujeres_cleaned['media'].dropna(), hombres_cleaned['media'].dropna())
    print(f"\nResultado de la prueba t (Hombres vs Mujeres): T-statistic = {t_stat}, P-value = {p_value}")

    # Si el P-value es menor que 0.05, podemos rechazar la hipótesis nula de que las medias son iguales

    # Modelo de regresión lineal: Plantear un modelo de regresión lineal sobre una variable dependiente
    # En este caso, vamos a realizar una regresión lineal para predecir la 'media' de mujeres en función de las demás variables
    X = mujeres_cleaned.drop(columns=['media'])
    y = mujeres_cleaned['media']

    # Asegurarse de que X no tenga NaN
    X = X.dropna()

    # Ajustar el modelo de regresión lineal
    if X.shape[0] > 0:  # Asegurarse de que haya al menos una muestra
        regressor = LinearRegression()
        regressor.fit(X, y)

        # Resultados del modelo de regresión
        print("\nResultados de la regresión lineal:")
        print(f"Coeficientes: {regressor.coef_}")
        print(f"Intercepto: {regressor.intercept_}")

        # Predicción para los datos de entrada
        y_pred = regressor.predict(X)

        # Mostrar algunas predicciones
        print(f"Predicciones para los primeros 5 valores: {y_pred[:5]}")
    else:
        print("No hay suficientes datos para la regresión.")
else:
    print("Los datos para la regresión están vacíos. Asegúrese de que los datos contengan suficiente información.")
