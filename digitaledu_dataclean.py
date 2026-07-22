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

# dropeamos lo innecesario:
df.drop(["id", "bdate", "has_photo", "langs"], axis=1, inplace=True)

print("-" * 40)
print("\nRESULTADOS")

df.info()
# Exportando para usar el csv limpio
# df.to_csv("cleaned/digitaledu_cleaned.csv")
