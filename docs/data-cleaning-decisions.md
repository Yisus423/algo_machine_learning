# Decisiones de Limpieza de Datos — DigitalEdu

## Criterios generales

Cada columna se evaluó con tres preguntas:
1. ¿Tiene valor predictivo? (correlación con `result` o ganancia de información)
2. ¿Está en formato que KNN pueda procesar? (KNN requiere datos numéricos, sin nulos, escalados)
3. ¿La calidad del dato justifica mantenerla?

---

## Columnas procesadas

### `sex`
- **Estado original**: `int64`, valores 1 (femenino) y 2 (masculino). Sin nulos.
- **Decisión**: Convertir a binario 0/1 (`sex - 1`).
- **Justificación**: Variable binaria clara. La codificación 0/1 es estándar para modelos de distancia como KNN. Correlación con `result`: **-0.69** (fuerte).

### `langs` → `declared_langs` + `num_langs`
- **Estado original**: String. Valores como `"Русский;English"` o `"False"` (1913 filas).
- **Decisión**:
  - `declared_langs`: 1 si el usuario declaró idiomas, 0 si es `"False"`.
  - `num_langs`: cantidad de idiomas contados con `len(x.split(";"))`.
  - Los `"False"` se convirtieron a NaN y se imputaron con la mediana del resto de los datos (**1**).
- **Justificación**: Dos columnas separan la información: "si respondió" de "cuántos idiomas". La mediana (1) coincide con la moda; imputar con 0 hubiera sido incorrecto (nadie habla 0 idiomas). Dropear el 23% de filas no se justificaba.

### `education_status` → 9 dummies
- **Estado original**: String, 9 valores únicos (ej: `"Alumnus (Specialist)"`, `"PhD"`).
- **Decisión**: One-hot encoding.
- **Justificación**: La documentación sugiere una escala ordinal, pero la correlación con `result` no sigue el orden esperado (PhD tiene la tasa de compra más baja). One-hot permite que KNN aprenda el peso independiente de cada categoría sin forzar una relación lineal inexistente.

### `education_form` → 4 dummies (con `dummy_na`)
- **Estado original**: String, 3 valores + 608 NaN.
- **Decisión**: One-hot encoding con `dummy_na=True`.
- **Justificación**: `pd.get_dummies(..., dummy_na=True)` captura los NaN como una categoría explícita, evitando imputar un valor que no conocemos.

### `occupation_type` → 3 dummies (con `dummy_na`)
- **Estado original**: String, 2 valores + 193 NaN.
- **Decisión**: One-hot encoding con `dummy_na=True`.
- **Justificación**: Mismo criterio que `education_form`. Las tasas de compra son muy similares (54.7% vs 53.1%), pero la columna no requiere mucho costo de mantenerla (solo 3 columnas extra).

### `city` → `city_freq`
- **Estado original**: String, 860 valores únicos + 582 NaN.
- **Decisión**: Frequency encoding: cada ciudad se reemplaza por su frecuencia de aparición. Los NaN se imputan con la mediana (1).
- **Justificación**: One-hot con 860 columnas es inviable para KNN (maldición de la dimensionalidad). Top-N (ej: top 20) solo cubría el 51% de los datos. Frequency encoding comprime la información en una sola columna numérica sin perder filas.

---

## Columnas eliminadas

### `id`
- **Motivo**: Identificador único, cero poder predictivo. Es el primer candidato a dropear en cualquier pipeline de ML.

### `bdate` (fecha de nacimiento)
- **Problema**: 14.3% NaN, 31.9% solo día/mes sin año, 53.9% con año completo. Entre los que tienen año, hay outliers de hasta 124 años.
- **Decisión**: Dropear.
- **Justificación**: Solo el 54% tiene año completo. Extraer edad implicaba perder el 46% de los datos o imputar valores con muy baja confianza por la calidad del dato.

### `has_photo`
- **Decisión**: Dropeada por el profesor en el script original. Se mantuvo la decisión.

### `life_main`, `people_main`
- **Problema**: 65% de las filas no tienen valor válido (mezcla de `"False"` string y 0). El 35% restante no correlaciona con `result` (correlación < 0.05).
- **Decisión**: Dropear.
- **Justificación**: No hay información útil que rescatar. Ni siquiera una columna binaria "respondió sí/no" mostró correlación significativa.

### `last_seen`
- **Problema**: Timestamp completo. Al extraer hora, mes y día de semana, la correlación con `result` fue 0.002, -0.007 y -0.000 respectivamente.
- **Decisión**: Dropear.
- **Justificación**: Ruido puro. No hay patrón temporal que prediga la compra.

### `occupation_name`
- **Problema**: 3922 valores únicos para 8193 filas. Nombres de instituciones en ruso. El valor más frecuente aparece solo 68 veces (0.8%).
- **Decisión**: Dropear.
- **Justificación**: Cardinalidad extrema. La información de tipo de ocupación ya está capturada por `occupation_type` (university/work).

### `career_start`, `career_end`
- **Problema**: 68% y 91% de `"False"` string respectivamente. Solo 704 filas tienen ambos datos. Duración calculada va de -68 a 69 años (datos inconsistentes). Correlación con `result` cercana a 0.
- **Decisión**: Dropear.
- **Justificación**: Demasiados datos faltantes para rescatar información confiable. La combinación de las dos columnas (duración de carrera) no correlaciona con el target y tiene outliers sin sentido.

---

## Columnas numéricas sin procesar (pasan directamente al modelo)

- **`has_mobile`**: Binaria (0/1), sin nulos.
- **`followers_count`**: Numérica, sin nulos. Escalada por StandardScaler en el modelo.
- **`graduation`**: Año de graduación, numérica. Escalada por StandardScaler.
- **`relation`**: Estado civil (0-8), numérica. El código 0 ya representa "no especificado". Escalada por StandardScaler.

---

## Resultado final

- **Columnas originales**: 20
- **Columnas después de limpieza**: 31 (incluyendo dummies y nuevas features)
- **Filas**: 8193 (ninguna eliminada)
- **Accuracy KNN (n=10)**: 80%

---

*Documentación generada para el curso. Todas las decisiones están basadas en análisis exploratorio previo, no en suposiciones.*
