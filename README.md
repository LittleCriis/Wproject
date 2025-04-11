Proyecto de Análisis de Brecha de Género en Ocupaciones Científicas y Tecnológicas
Este proyecto se centra en el análisis de la brecha de género en diferentes ocupaciones científicas y tecnológicas utilizando datos del Instituto Nacional de Estadística (INE). El objetivo es analizar las diferencias entre la cantidad de hombres y mujeres empleados en estos sectores a lo largo de los años, con especial enfoque en ocupaciones como científicos, matemáticos y tecnólogos.

Contenido
Descripción del Proyecto

Tecnologías Utilizadas

Pasos Realizados

Análisis Estadísticos

Visualización en Power BI

Conclusiones

Requerimientos

1. Descripción del Proyecto
En este proyecto, se analizan los datos proporcionados por el INE sobre la distribución de trabajadores por género en diferentes sectores ocupacionales a lo largo de los años. Se hace especial énfasis en el análisis de la brecha de género en puestos científicos, tecnológicos, matemáticos y otros relacionados. El proyecto se desarrolló utilizando herramientas como Python, Pandas y Power BI para realizar un análisis exploratorio y crear visualizaciones interactivas.

2. Tecnologías Utilizadas
Las tecnologías y librerías utilizadas en este proyecto incluyen:

Python: Lenguaje de programación utilizado para el procesamiento y análisis de datos.

Pandas: Para manipular y limpiar los datos en formato CSV.

Numpy: Para realizar operaciones numéricas y trabajar con matrices.

Matplotlib / Seaborn: Para la creación de gráficos de visualización exploratoria, como histogramas, boxplots y gráficos de barras.

Scipy / Statsmodels: Para realizar pruebas estadísticas, como la prueba t de Student para comparar la media entre los géneros en diferentes puestos.

Power BI: Herramienta de visualización de datos que permite crear gráficos interactivos y dashboards.

DAX (Data Analysis Expressions): Lenguaje utilizado en Power BI para crear cálculos y medidas personalizadas.

3. Pasos Realizados
3.1 Obtención de Datos
Los datos utilizados fueron descargados del sitio web del Instituto Nacional de Estadística (INE) en formato CSV. Los archivos contienen información sobre las ocupaciones de hombres y mujeres a lo largo de los años, desglosados por distintos puestos de trabajo.

3.2 Limpieza de Datos
Imputación de valores nulos: Se imputaron los valores nulos en los datos para evitar que los cálculos se vieran sesgados o incompletos.

Conversión de datos: Se transformaron las columnas de los trimestres en un solo campo para cada año, sumando los trimestres correspondientes.

Filtrado de valores: Se eliminaron filas innecesarias, como la fila de "Total", que no representaba datos relevantes para el análisis.

3.3 Análisis Exploratorio
Se realizaron análisis exploratorios para observar las tendencias en la distribución de los hombres y mujeres en las ocupaciones científicas y tecnológicas a lo largo de los años. Se usaron herramientas como Seaborn y Matplotlib para crear gráficos de barras, boxplots y análisis de tendencias.

3.4 Generación de Medidas
Para facilitar el análisis en Power BI, se crearon medidas de totalización para los datos de hombres y mujeres, sumando los valores anuales de cada puesto:

Total Hombres = 
SUM('Hombres'[2011]) + 
SUM('Hombres'[2012]) + 
SUM('Hombres'[2013]) + 
... (hasta 2024)

3.5 Visualización en Power BI
Se importaron los datos a Power BI, y se crearon diferentes visualizaciones, entre ellas:

Segmentación: para elegir algunos puestos en concreto

Gráfico circular: Para mostrar información sobre los porcentajes de hombres y mujeres en distintos puestos.

Gráfico de barras: Para comparar la distribución de hombres y mujeres en los distintos puestos por año.

Líneas de tiempo: Para mostrar las tendencias de la brecha de género a lo largo de los años.

Filtros dinámicos: Para seleccionar los puestos y los años, y filtrar los resultados de manera más específica.

4. Análisis Estadísticos
4.1 Prueba T para Hombres vs Mujeres
Se realizó una prueba t de Student para comparar si había diferencias significativas en los salarios o el número de empleados entre hombres y mujeres en las diferentes ocupaciones a lo largo de los años. Los resultados indicaron si las diferencias observadas eran estadísticamente significativas.

Resultados:

T-statistic y P-value fueron calculados para cada comparación entre hombres y mujeres, determinando la significancia de las diferencias en los valores de los puestos seleccionados.

4.2 Estadísticas Descriptivas
Se calcularon las medias, medianas, desviaciones estándar y otros estadísticos descriptivos para cada puesto a lo largo de los años, permitiendo entender mejor la distribución de los datos y las diferencias entre géneros en cada puesto.

4.3 Visualización de la Diferencia de Género
Se visualizaron las diferencias utilizando gráficos de líneas y de barras, comparando la cantidad de hombres y mujeres empleados en los sectores seleccionados en diferentes años. Estas visualizaciones ayudaron a identificar tendencias y patrones a lo largo del tiempo.

5. Visualización en Power BI
5.1 Dashboard Principal
El dashboard principal permite a los usuarios explorar la brecha de género en las ocupaciones científicas y tecnológicas de manera interactiva. Los filtros permiten seleccionar los puestos y los años específicos, y las visualizaciones muestran cómo ha cambiado la representación de hombres y mujeres en estos sectores a lo largo del tiempo.

5.2 Gráficos y Medidas
Se utilizaron gráficos de barras y líneas para mostrar las distribuciones por género en cada puesto y cómo han evolucionado a lo largo de los años. También se implementaron filtros de jerarquía para que los usuarios pudieran profundizar en diferentes categorías de puestos.

6. Conclusiones
Este análisis muestra que, aunque ha habido avances en la inclusión de mujeres en sectores científicos y tecnológicos, todavía queda un largo camino por recorrer. Las políticas de igualdad de género deben continuar enfocándose en eliminar las barreras que impiden una representación equitativa en estos campos.