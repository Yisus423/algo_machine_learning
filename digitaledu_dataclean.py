import pandas as pd

# Paso 1. Cargar y limpiar los datos
df = pd.read_csv("data/digitaledu.csv")
df.info()

# -------langs--------------------
# Creamos la columna declared_langs
# 1: Respondió la pregunta
# 0: No lo hizo
df["declared_langs"] = (df["langs"] != "False").astype(int)

# Creamos la columna num_langs
# sí respondió la pregunta se aplica la lambda para obtener el num
# si no (NaN) lo rellenamos con la mediana para no ensuciar datos
df["num_langs"] = (
    df["langs"]
    .apply(lambda x: len(x.split(";")))
    .where(df["declared_langs"] == 1)  # sí la condicion es falsa convierte a NaN
    .fillna(1)  # la mediana de num_langs
)

# ---------education_status-------
# convertimos los strings de education_status a 9 dummies y los concatenamos al dataframe
df = pd.concat([df, pd.get_dummies(df["education_status"], prefix="edu")], axis=1)

# dropeamos lo innecesario:
df.drop(["id", "bdate", "has_photo", "langs", "education_status"], axis=1, inplace=True)

print("-" * 40)
print("\nRESULTADOS")

df.info()
# Exportando para usar el csv limpio
# df.to_csv("cleaned/digitaledu_cleaned.csv")
