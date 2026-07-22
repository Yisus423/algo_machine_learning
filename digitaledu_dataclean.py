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
temp_langs = (
    df["langs"]
    .apply(lambda x: len(x.split(";")))
    .where(df["declared_langs"] == 1)  # sí la condicion es falsa convierte a NaN
)
median_langs = temp_langs.median()
df["num_langs"] = temp_langs.fillna(median_langs)

# ---------education_status-------
# convertimos los strings de education_status a 9 dummies y los concatenamos al dataframe
df = pd.concat([df, pd.get_dummies(df["education_status"], prefix="edu")], axis=1)

# ----------education_form---------
# Misma lógica que el anterior, 4 columnas dummies incluyendo las NaN
df = pd.concat(
    [df, pd.get_dummies(df["education_form"], prefix="form", dummy_na=True)], axis=1
)


# ----------ocupation_type---------
# Misma lógica... 3 dummies

df = pd.concat(
    [df, pd.get_dummies(df["occupation_type"], prefix="occ", dummy_na=True)], axis=1
)

# ---------sex--------------------
# Convertimos los valores de 1/2 a binario (1/0)
# simplemente le restamos 1 a todos
df["sex"] = df["sex"] - 1  # 1 = male, 0 = female

# dropeamos lo innecesario:
INNECESARY_COLUMNS = [
    "id",
    "bdate",
    "has_photo",
    "langs",
    "education_status",
    "life_main",
    "people_main",
    "education_form",
    "occupation_type",
]
df.drop(INNECESARY_COLUMNS, axis=1, inplace=True)

print("-" * 40)
print("\nRESULTADOS")

df.info()
# Exportando para usar el csv limpio
# df.to_csv("cleaned/digitaledu_cleaned.csv")
