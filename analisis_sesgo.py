import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, recall_score

# ==============================================================================
# UNIDAD 1.1: INTRODUCCIÓN Y MARCO DE ANALÍTICA PREDICTIVA
# Objetivo: Cargar datos y entender el problema de negocio (Detección de Riesgo)
# ==============================================================================

print("--- INICIO UNIDAD 1.1: CARGA DE DATOS ---")

# 1. Carga del Dataset 
df = pd.read_csv('german_credit_data_ibm.csv')

# 2. Entender el Problema: ¿Qué estamos prediciendo?
# La variable 'Risk' es nuestra 'Y' (Target).
# Risk = 'Risk' (Mal pagador) vs 'No Risk' (Buen pagador)
print(f"Dimensiones del dataset: {df.shape}")
print("\nDistribución de la variable objetivo (Risk):")
print(df['Risk'].value_counts(normalize=True))

# Siempre verifico la proporción.
# Si 'Risk' es muy bajo (<10%), tendríamos un problema de desbalance severo.

# ==============================================================================
# UNIDAD 1.2: PREPARACIÓN DE DATOS (DATA PREPARATION)
# Objetivo: Limpiar, codificar y dividir datos para el modelado.
# ==============================================================================

print("\n--- INICIO UNIDAD 1.2: PREPARACIÓN DE DATOS ---")
print("--- Primeras filas del dataset sin limpiar CustomerID ---")
print(df.head())

# 1. Eliminamos identificadores que no predicen nada (CustomerID)
if 'CustomerID' in df.columns:
    df = df.drop('CustomerID', axis=1)
    print("--- Primeras filas del dataset limpiado CustomerID ---")
    print(df.head())

# 2. Codificación de la Variable Objetivo (Label Encoding)
# Convertimos 'Risk' a 1 y 'No Risk' a 0 para que el modelo lo entienda.
# Esto es vital: Queremos predecir el "Riesgo" (1).
df['Target'] = df['Risk'].apply(lambda x: 1 if x == 'Risk' else 0)
y = df['Target']
X = df.drop(['Risk', 'Target'], axis=1) # Las features son todo menos el target
print("--- Primeras filas del dataset limpiado CustomerID y Target ---")
print(df.head())

# 3. Separación de Variables Numéricas y Categóricas
numeric_features = ['LoanDuration', 'LoanAmount', 'InstallmentPercent', 'Age', 
                    'CurrentResidenceDuration', 'ExistingCreditsCount', 'Dependents']
categorical_features = ['CheckingStatus', 'CreditHistory', 'LoanPurpose', 'ExistingSavings', 
                        'EmploymentDuration', 'Sex', 'OthersOnLoan', 'OwnsProperty', 
                        'InstallmentPlans', 'Housing', 'Job', 'Telephone', 'ForeignWorker']

# 4. Split de Datos (Train / Test)
# Como vimos en Clase 1: Usamos 80% para entrenar y 20% para evaluar (Hold-out)
# stratify=y asegura que la proporción de Riesgo se mantenga en ambos sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Datos de Entrenamiento: {X_train.shape}")
print(f"Datos de Prueba: {X_test.shape}")

# 5. Pipeline de Preprocesamiento
# Escalamos numéricos y hacemos One-Hot Encoding a categóricos automáticamente
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# ==============================================================================
# UNIDAD 1.3: MODELADO Y MÉTRICAS DE DESEMPEÑO
# Objetivo: Entrenar un modelo base y medir su éxito (Recall, Precision).
# ==============================================================================

print("\n--- INICIO UNIDAD 1.3: MODELADO Y MÉTRICAS ---")

# 1. Definición del Modelo (Regresión Logística)
# Usamos un modelo simple primero para tener un 'Baseline'.
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# 2. Entrenamiento (Fit)
pipeline.fit(X_train, y_train)

# 3. Predicción
y_pred = pipeline.predict(X_test)

# 4. Evaluación (Lo crucial de la Unidad 1.3)
print("\nMatriz de Confusión:")
# Nos dice: ¿Cuántos riesgos reales detectamos? (Verdaderos Positivos)
print(confusion_matrix(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Métrica Clave: Recall (Sensibilidad)
# En Riesgo Crediticio, nos importa el RECALL de la clase 1.
# ¿Por qué? Porque es más costoso no detectar un riesgo (falso negativo) que rechazar a un bueno.
recall = recall_score(y_test, y_pred)
print(f"Recall del modelo (Capacidad de detectar Riesgo): {recall:.2f}")

print("\n--- FIN DE LA FASE DE MODELADO (INICIANDO ANÁLISIS ÉTICO 1.4) ---")
print("-" * 60)

# ==============================================================================
# UNIDAD 1.4: Análisis de Sesgo
# Objetivo: Entrenar un modelo base y medir su éxito (Recall, Precision).
# ==============================================================================

print("\n--- INICIO UNIDAD 1.4: ANÁLISIS DE SESGO ---")

# 1. Cargamos el dataset
# df = pd.read_csv('german_credit_data_ibm.csv')

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