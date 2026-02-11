import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargamos el dataset
# Nota: el nombre del archivo  correcto (ej. 'german_credit_data_ibm.csv')
df = pd.read_csv('german_credit_data_ibm.csv')

# 2. Inspección rápida de las primeras filas
print("--- Primeras filas del dataset ---")
print(df.head())

# 3. Identificación de variables sensibles
# Vamos a ver cuántos hombres y mujeres hay en el dataset
print("\n--- Distribución por Género ---")
print(df['Sex'].value_counts())

# 4. Análisis de Sesgo: ¿Hay una relación desproporcionada entre Sexo y Riesgo?
# Creamos una tabla cruzada para ver el porcentaje de 'Riesgo Malo' por género
cross_tab = pd.crosstab(df['Sex'], df['Risk'], normalize='index') * 100
print("\n--- Porcentaje de Riesgo por Género (%) ---")
print(cross_tab)

# 5. Visualización para el reporte de gobernanza
plt.figure(figsize=(8, 5))
sns.countplot(x='Sex', hue='Risk', data=df, palette='magma')
plt.title('Distribución de Riesgo Crediticio por Género')
plt.xlabel('Género')
plt.ylabel('Cantidad de Clientes')
plt.savefig('analisis_sesgo_genero.png') # Guardamos el artefecto para tu reporte
# plt.show() # Comentamos esto momentáneamente para que el programa siga y muestre el de edad también.

# 6. Agrupación por Edades
# A veces la edad exacta es difícil de analizar, así que hacemos grupos:
# 19-30 (Joven), 31-50 (Adulto), 51+ (Mayor)
bins = [18, 30, 50, 100] 
labels = ['Joven (19-30)', 'Adulto (31-50)', 'Mayor (51+)']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

# 7. Ver porcentajes por edad
print("\n--- Riesgo por Grupo de Edad (%) ---")
print(pd.crosstab(df['AgeGroup'], df['Risk'], normalize='index') * 100)

# 8. Gráfico de Edad
plt.figure(figsize=(8, 5))
sns.countplot(x='AgeGroup', hue='Risk', data=df, palette='viridis') # 'viridis' es otra paleta de colores
plt.title('Riesgo Crediticio por Grupo de Edad')
plt.xlabel('Edad')
plt.ylabel('Cantidad')
plt.savefig('analisis_sesgo_edad.png')
plt.show()