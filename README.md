# Machine Learning - Prácticas

Espacio de prácticas de Machine Learning con **pandas** y **scikit-learn**. El proyecto cubre el flujo completo de un modelo de clasificación: carga de datos, limpieza, entrenamiento y evaluación.

---

## Arquitectura de carpetas

```
machine_learning/
├── data/                        # Datos crudos (originales)
│   ├── titanic.csv
│   └── digitaledu.csv
├── cleaned/                     # Datos limpios (resultado de la limpieza)
│   └── titanic_cleaned.csv
├── explanations/                # Material de explicaciones (futuro)
├── titanic_dataclean.py         # Script de limpieza de datos del Titanic
├── digitaledu_dataclean.py      # Script de limpieza de datos de DigitalEdu
├── model_titanic.py             # Modelo KNN para predecir supervivencia en Titanic
├── model_digitaledu.py          # Modelo KNN para predecir en DigitalEdu
├── pyproject.toml               # Dependencias del proyecto (pandas, sklearn, streamlit, matplotlib)
└── README.md
```

### Flujo de trabajo

```
data/*.csv  →  *_dataclean.py  →  cleaned/*.csv  →  model_*.py  →  Evaluación
 (crudos)        (limpieza)          (limpios)         (modelo)       (resultados)
```

---

## Funciones de pandas utilizadas

### Carga y exportación de datos

| Función | Descripción | Ejemplo |
|---------|-------------|---------|
| `pd.read_csv()` | Lee un archivo CSV y lo convierte en un DataFrame | `df = pd.read_csv('data/titanic.csv')` |
| `df.to_csv()` | Exporta un DataFrame a un archivo CSV | `df.to_csv('cleaned/titanic_cleaned.csv')` |
| `df.info()` | Muestra el resumen de columnas, tipos de datos y valores nulos | `df.info()` |

### Manipulación de datos

| Función | Descripción | Ejemplo |
|---------|-------------|---------|
| `df.drop()` | Elimina columnas o filas por etiqueta | `df.drop(['Name', 'Ticket'], axis=1, inplace=True)` |
| `df.drop()` con `axis=1` | Elimina **columnas** (axis=0 elimina filas) | `df.drop('Embarked', axis=1, inplace=True)` |
| `df.fillna()` | Rellena valores nulos con un valor específico | `df['Embarked'] = df['Embarked'].fillna('S')` |
| `df['col'].median()` | Calcula la mediana de una columna | `df[df['Pclass'] == 1]['Age'].median()` |
| `df.apply()` | Aplica una función a cada fila o columna | `df['Age'] = df.apply(fill_age, axis=1)` |
| `df['col'].apply()` | Aplica una función elemento a elemento en una columna | `df['Sex'] = df['Sex'].apply(fill_sex)` |
| `pd.isnull()` | Verifica si un valor es nulo (NaN) | `if pd.isnull(row['Age']):` |
| `df['col'].value_counts()` | Cuenta la frecuencia de cada valor único en una columna | `df['occupation_type'].value_counts()` |
| `df.drop()` con `inplace=True` | Modifica el DataFrame directamente sin reasignarlo | `df.drop(columns, axis=1, inplace=True)` |

### Transformación de categorías

| Función | Descripción | Ejemplo |
|---------|-------------|---------|
| `pd.get_dummies()` | Convierte columnas categóricas en variables numéricas (one-hot encoding). Crea una columna por cada valor único con 0 o 1 | `pd.get_dummies(df, columns=['Sex', 'Embarked'], dtype=int)` |
| `pd.get_dummies()` con `dtype=int` | Especifica el tipo de dato de las nuevas columnas (por defecto es bool) | `pd.get_dummies(df, columns=['col'], dtype=int)` |

> **¿Cuándo usarlo?** Cuando el modelo recibe columnas con texto (ej: "male"/"female", "S"/"C"/"Q"). Los modelos de scikit-learn solo aceptan números, así que `get_dummies` las convierte en columnas binarias.

### Filtrado

| Función | Descripción | Ejemplo |
|---------|-------------|---------|
| `df[df['col'] == valor]` | Filtra filas donde la columna cumple una condición | `df[df['Pclass'] == 1]` |

---

## Funciones de scikit-learn utilizadas

### Preprocesamiento

| Función | Módulo | Descripción |
|---------|--------|-------------|
| `train_test_split()` | `sklearn.model_selection` | Divide los datos en conjuntos de entrenamiento (75%) y prueba (25%) |
| `StandardScaler()` | `sklearn.preprocessing` | Escala las features para que tengan media=0 y desviación estándar=1 |
| `sc.fit_transform()` | `sklearn.preprocessing` | Calcula media/std del set de entrenamiento y lo transforma |
| `sc.transform()` | `sklearn.preprocessing` | Aplica la misma transformación al set de prueba (sin recalcular) |

### Modelo

| Función | Módulo | Descripción |
|---------|--------|-------------|
| `KNeighborsClassifier()` | `sklearn.neighbors` | Clasificador K-Vecinos Más Cercanos (KNN) |
| `classifier.fit()` | - | Entrena el modelo con los datos de entrenamiento |
| `classifier.predict()` | - | Genera predicciones sobre el set de prueba |

### Métricas de evaluación

| Función | Módulo | Descripción |
|---------|--------|-------------|
| `accuracy_score()` | `sklearn.metrics` | Calcula el porcentaje de predicciones correctas |
| `confusion_matrix()` | `sklearn.metrics` | Retorna una matriz 2x2 con TP, TN, FP, FN |

#### Matriz de confusión

```
                  Predicho
                  +      -
Real    +    [  TP  |  FN  ]
        -    [  FP  |  TN  ]

TP = Verdaderos positivos (sobrevivió y se predijo correctamente)
TN = Verdaderos negativos (murió y se predijo correctamente)
FP = Falsos positivos (murió pero se predijo que sobreviviría)
FN = Falsos negativos (sobrevivió pero se predijo que moriría)
```

---

## Cómo ejecutar

```bash
# 1. Instalar dependencias
pip install -e .

# 2. Limpiar datos del Titanic
python titanic_dataclean.py

# 3. Entrenar y evaluar el modelo
python model_titanic.py
```

---

## Dependencias

- **pandas** >= 3.0.3 - Manipulación de datos
- **scikit-learn** >= 1.9.0 - Modelos de Machine Learning
- **matplotlib** >= 3.11.0 - Gráficos (futuro)
- **streamlit** >= 1.57.0 - Interfaz web (futuro)
