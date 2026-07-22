import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

N_NEIGHBORS = 10
TEST_SIZE = 0.25


def train_testing(x, y):
    """--- División en entrenamiento y prueba ---"""
    # test_size=0.25: el 25% de los datos se reserva para evaluar el modelo
    # El 75% restante se usa para entrenar (evita overfitting al evaluar con datos "vistos")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE)

    # TIP: se puede jugar con el tamaño de prueba

    """ --- Escalado de features (Feature Scaling) --- """
    # Si un dato tiene valores mucho mayores que otra dominará la distancia.
    # StandardScaler estandariza cada feature: z = (x - media) / desviación_estándar
    sc = StandardScaler()
    # fit_transform: calcula media y std del set de ENTRENAMIENTO y transforma
    x_train = sc.fit_transform(x_train)
    # transform: aplica la misma transformación al set de PRUEBA (no recalcula media/std)
    x_test = sc.transform(x_test)

    """ --- Entrenamiento del modelo KNN --- """
    # KNN clasifica un punto según la clase mayoritaria de sus K vecinos más cercanos
    # n_neighbors=5: se consideran los 5 vecinos más cercanos para decidir la clase
    classifier = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    # fit: el modelo "aprende" la relación features -> objetivo a partir de los datos
    classifier.fit(x_train, y_train)

    # TIP: se puede jugar con el número de vecinos

    """ --- Evaluación del modelo --- """
    # predict: el modelo predice la clase para cada muestra del set de prueba
    y_pred = classifier.predict(x_test)
    # accuracy_score: proporción de predicciones correctas (0.0 a 1.0, se multiplica por 100)
    print(
        f"Porcentaje de resultados predichos correctamente: {round(accuracy_score(y_test, y_pred) * 100, 2)}%"
    )
    # Ej: Porcentaje de resultados predichos correctamente: 79½

    print("Confusion matrix:")

    """ --- Desempaquetando en las variables las 2 listas --- """
    # la funcion confusion_matrix() retorna 2 listas con los resultados
    positives, negatives = confusion_matrix(y_test, y_pred)
    # Desempaquetando las 2 listas en sus pares de datos
    true_positive, false_positive = positives
    false_negative, true_negative = negatives

    print(f"""
    RESULTADOS:
    {"-" * 40}
    TN: {true_negative} verdaderos negativos (murió y se predijo muerte)
    TP: {true_positive} verdaderos positivos (sobrevivió y se predijo supervivencia)

    FP: {false_positive} falsos positivos (murió y se predijo supervivencia)
    FN: {false_negative} falsos negativos (sobrevivió y se predijo muerte)
    {"-" * 40}

    """)


# Carga de datos
df = pd.read_csv("cleaned/digitaledu_cleaned.csv")

x = df.drop("result", axis=1)  # todas las columnas excepto result
y = df["result"]  # solo la columna target
train_testing(x, y)
