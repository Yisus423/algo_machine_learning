import pandas as pd 

# Paso 1. Cargar y limpiar los datos
df = pd.read_csv('data/digitaledu.csv')
df.info()

df.drop(["id","bdate", "has_photo"], axis=1, inplace= True)

print("-"*40)
print(df["occupation_type"].value_counts())

print("-"*40)
print("\nRESULTADOS")

df.info()
# Exportando para usar el csv limpio
# df.to_csv("cleaned/digitaledu_cleaned.csv")