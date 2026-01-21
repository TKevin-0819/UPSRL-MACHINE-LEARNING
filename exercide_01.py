# 1. Preparación del entorno
#    - Usa la librería pandas.
#    - Usa la librería numpy.
#    - Usa la librería scipy.
#    Objetivo: Tener listo el entorno de trabajo.
#
# 2. Cargar el archivo CSV
#    - Usa pandas para leer el archivo "alumnos.csv".
#    - Muestra las primeras filas de la tabla.
#    Objetivo: Familiarizarse con la estructura del DataFrame.
#
# 3. Explorar los datos
#    - Usa pandas para revisar la información de columnas y tipos de datos.
#    - Usa pandas para obtener estadísticas básicas de las columnas numéricas.
#    Objetivo: Identificar qué tipo de información contiene cada columna.
#
# 4. Selección y filtrado
#    - Usa pandas para seleccionar la columna de nombres.
#    - Usa pandas para filtrar los alumnos cuyo promedio sea mayor a 9.
#    Objetivo: Aprender a extraer subconjuntos de datos.
#
# 5. Operaciones con NumPy
#    - Usa numpy para convertir la columna de edades en un arreglo.
#    - Usa numpy para calcular medidas estadísticas básicas (media, desviación estándar).
#    Objetivo: Practicar cálculos numéricos con NumPy.
#
# 6. Agregar y transformar columnas
#    - Usa pandas junto con numpy para crear una nueva columna que indique
#      si el alumno es "Excelente" o "Regular" según su promedio.
#    Objetivo: Aprender a enriquecer los datos con nuevas variables.
#
# 7. Guardar resultados
#    - Usa pandas para exportar el DataFrame modificado a un nuevo archivo CSV.
#    Objetivo: Practicar la escritura de datos en CSV.
#
# 8. Operaciones con SciPy
#    - Usa scipy para realizar pruebas estadísticas sobre los datos de los alumnos.
#    - Ejemplo de aplicación: comparar edades o promedios con una distribución teórica,
#      o verificar si hay diferencias significativas entre grupos.
#    Objetivo: Introducir el uso de herramientas estadísticas más avanzadas.
#
# 9. Preguntas interpretativas
#    1. ¿Qué diferencia hay entre un DataFrame y un arreglo de NumPy?
#    2. ¿Por qué es útil explorar los datos antes de hacer cálculos?
#    3. ¿Qué aporta SciPy respecto a pandas y numpy en el análisis de datos?
#    4. ¿Cómo podrías usar estas herramientas para analizar el rendimiento de toda una clase?

import os
import pandas as pd
import numpy as np
import scipy as sp
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

#windows: inputs\estudiantes.csv
#Linux: inputs/estudiantes.csv
SOURCE_FILE = os.path.join(".", "inputs", "estudiantes.csv")

print(SOURCE_FILE)

#despliego solamente de los primeros n estudiantes.
estudiantes = pd.read_csv(SOURCE_FILE)
print(estudiantes.head(n=3))
print()
#Desplegamos columnas de nuestro data frame
print(estudiantes.columns)



    #print(estudiantes["matricula"])
    #print(estudiantes["promedio"])
#PASO NUMERO 3
# Iteramos las columnas de nuestro dataframe para visualizar el tipo de datos
for col in estudiantes.columns:
    
    data_type = estudiantes[col].dtype
    print("Nombre de columna:",col)
    print("Tipo de dato:", data_type)
    
    
    if data_type == int or data_type == float :
        print("metricas:")
        print("-", "Media", estudiantes[col].mean())
        print("-", "Mediana", estudiantes[col].median())
        print("-", "Maximo", estudiantes[col].max())
        print("-", "Minimo", estudiantes[col].min())
        
    print("-----==---")
    
# 4. Selección y filtrado
#    - Usa pandas para seleccionar la columna de nombres.
#    - Usa pandas para filtrar los alumnos cuyo promedio sea mayor a 9.
#    Objetivo: Aprender a extraer subconjuntos de datos.

nombres = estudiantes["nombre"]
mejores = estudiantes[estudiantes["promedio"] > 9]

apellidos_con_l = estudiantes[estudiantes["apellido"].str.startswith ("G")]



# 5. Operaciones con NumPy
#    - Usa numpy para convertir la columna de edades en un arreglo.
#    - Usa numpy para calcular medidas estadísticas básicas (media, desviación estándar).
#    Objetivo: Practicar cálculos numéricos con NumPy.


#conertimos columna de edades en array de numpy

edades_pandas = estudiantes["edad"].to_numpy()

edades_numpy = np.array(estudiantes["edad"])

print("-", "media", edades_numpy.mean())
print("-", "desviación estandar",edades_numpy.std())
print("-", "maximos",edades_numpy.max())
print("-", "minimo",edades_numpy.min())

#creamos nueva columna con una evaluacion del alumno conforme a su promedio
estudiantes["opinion"] = np.where(estudiantes["promedio"] > 9, "Excelente", "regular")


#windows: inputs\estudiantes.csv
#Linux: inputs/estudiantes.csv
OUTPUT_FILE = os.path.join(".", "out", "estudiantes.csv")
OUTPUT_DIR = os.path.dirname(OUTPUT_FILE)


#checamos si esta vacio el archivo
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.mkdir(OUTPUT_DIR)


estudiantes.to_csv(OUTPUT_FILE)

edades_pandas = estudiantes["edad"].to_numpy()
cuatrimestres_pandas = estudiantes["cuatrimestre"].to_numpy()
promedios_pandas = estudiantes["edad"].to_numpy()


# 8. Operaciones con SciPy
#    - Usa scipy para realizar pruebas estadísticas sobre los datos de los alumnos.
#    - Ejemplo de aplicación: comparar edades o promedios con una distribución teórica,
#      o verificar si hay diferencias significativas entre grupos.
#    Objetivo: Introducir el uso de herramientas estadísticas más avanzadas.

descripcion = stats.describe(edades_pandas)
print("statistics with sciPy:")
print(f"mean: {descripcion.mean}")

test_edad= stats.ttest_1samp(edades_pandas, 20)
print("Edad vs 20: ")
print(test_edad. _statistic_np)
print(test_edad._standard_error)
print(test_edad._estimate)
print("- pvalue")
print("test")
#t_prom, p_prom = stats.ttest_1samp(promedios_pandas, 7.5)

#shapiro -- (statistic, pvalue)
p_edad_norm = stats.shapiro(edades_pandas)[1]
p_prom_norm = stats.shapiro(promedios_pandas)[1]

print("Normalidad Edad vs 20 -> = ", p_edad_norm)
print('Normalidad Promedio vs -> 7.5 p =', p_prom_norm)

plt.figure(figsize=(10, 4))
plt.title("Mi primer dibujo")
plt.xlabel("Cuatrimestre")
plt.ylabel("Promedio")

sns.boxplot(data=estudiantes, x="cuatrimestre", y="promedio")
OUTPUT_FIG = os.path.join(OUTPUT_DIR, "figuree.png")

#Guardar y desplegar gráfico
plt.savefig(OUTPUT_FIG)  # guarda la gráfica en un archivo
plt.show()