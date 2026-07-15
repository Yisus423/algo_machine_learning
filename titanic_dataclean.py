import pandas as pd 
 
# pd.get_dummies(df, columns=['<column>'], dtype=int)

# Paso 1. Cargar y limpiar los datos
df = pd.read_csv('data/titanic.csv')
df.info()

df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
df['Embarked'] =  df['Embarked'].fillna('S')
df.drop('Embarked', axis = 1, inplace = True)
 
# Obteniendo la mediana de la edad de cada clase del barco
age_1 = df[df['Pclass'] == 1]['Age'].median()
age_2 = df[df['Pclass'] == 2]['Age'].median()
age_3 = df[df['Pclass'] == 3]['Age'].median()

# llenando con la mediana las edades que estén vacías, segun la clase en que se encuentren.
def fill_age(row):
   if pd.isnull(row['Age']):
       if row['Pclass'] == 1:
           return age_1
       if row['Pclass'] == 2:
           return age_2
       return age_3
   return row['Age']

# Aplicando la función sobre las filas (por eso el "axis = 1", de este modo se puede obtener el acceso a los datos de esa fila)
df['Age'] = df.apply(fill_age, axis = 1)
 
# Cambiando el valor del genero de 0 a 1 (0 = mujer y 1 = hombre)
def fill_sex(sex):
    if sex == 'male':
        return 1
    return 0

# Aplicando la función de reemplazo por 0 y 1 al género
df['Sex'] = df['Sex'].apply(fill_sex)

print("-"*40)
print("\nRESULTADOS")

df.info()
#Exportando para usar el csv limpio
df.to_csv("cleaned/titanic_cleaned.csv")