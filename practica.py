import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df= r"C:\Users\patri\Downloads\encuesta_ejemplo.csv" 
df = pd.read_csv(df)
print(df.head())

print(df.columns)

grupo_sexo_estrato = df.groupby(["Sexo", "Estrato"])

# Ver todas las combinaciones de grupos
print(grupo_sexo_estrato.groups.keys())

# Ver solo un grupo específico
print(grupo_sexo_estrato.get_group(("Masculino", "Medio")))

agrupamiento = df.groupby(["Edad", "Nivel Educativo"])[["Edad", "Nivel Educativo"]]
print(agrupamiento.head())

ejemplo_agrupamiento = df.groupby(["Estrato", "Nivel Educativo"])[["Estrato", "Nivel Educativo"]]
print(ejemplo_agrupamiento.head())

#Aca chat sugirió chi cuadrado porque son variables categóricas, pero terminé recategorizando
pd.crosstab(df["Estrato"], df["Nivel Educativo"])
df['estrato_num'] = df['Estrato'].map({"Bajo": 1, "Medio": 2, "Alto": 3})
df['nivel_num'] = df['Nivel Educativo'].map({"Primario": 1, "Secundario": 2, "Terciario": 3, "Universitario": 4})


# # Correlación de Pearson o Spearman directamente con pandas
corr_spearman = df[['estrato_num', 'nivel_num']].corr(method='spearman')
print(corr_spearman)

corr = df[['estrato_num', 'nivel_num']].corr(method='spearman') 
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlación entre Estrato y Nivel Educativo")
plt.show()
#Solo correlación


tabla_pct = pd.crosstab(df['Estrato'], df['Nivel Educativo'], normalize='index') * 100
sns.heatmap(tabla_pct, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Porcentaje de Nivel Educativo por Estrato")
plt.show()

# Definimos los límites de cada grupo
bins = [17, 24, 34, 49, 64, 120]  # límites de edad
labels = ['18-24', '25-34', '35-49', '50-64', '65+']  # nombres de las categorías

# Creamos la nueva columna categórica
df['Rango_Edad'] = pd.cut(df['Edad'], bins=bins, labels=labels, right=True)

# Verificamos el resultado
print(df[['Edad', 'Rango_Edad']].head())


df.groupby("Rango_Edad")

pd.crosstab(df["Rango_Edad"], df["Nivel Educativo"])

 df["Rango_Edad"].value_counts().sort_index().plot(kind="bar")

#Empiezo a ver imagen del candidato
 print(df[['Imagen del Candidato']])

 print(df[['Imagen del Candidato']].dtypes)
df['Imagen del Candidato'] = pd.to_numeric(df['Imagen del Candidato'], errors='coerce')

bins = [0, 20, 40, 80, 100]  # 0-25, 26-50, 51-75, 76-100
labels = ['Mala', 'Regular', 'Buena', 'Excelente']

df['Imagen_Rango'] = pd.cut(df['Imagen del Candidato'], bins=bins, labels=labels, right=True)

# Verificamos
print(df[['Imagen del Candidato', 'Imagen_Rango']])

orden = ['Mala', 'Regular', 'Buena', 'Excelente']

#Aca el tema es que ordené de menor a mayor, pero no se hace visual
df['Imagen_Rango'] = pd.Categorical(df['Imagen_Rango'], categories=orden, ordered=True)

# Verificar
print(df[['Imagen del Candidato', 'Imagen_Rango']].head(10))

tabla = pd.crosstab(df['Estrato'], df['Imagen_Rango']) # Aca la acomodó 
print(tabla)

tabla_pct = pd.crosstab(df['Estrato'], df['Imagen_Rango'], normalize='index') * 100

sns.heatmap(tabla_pct, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Porcentaje de Imagen del Candidato por Estrato")
plt.show()


# Cantidad de personas por categoría
print(df['Imagen_Rango'].value_counts())

# Porcentaje de personas por categoría
print(df['Imagen_Rango'].value_counts(normalize=True) * 100)

# Cruzado con Estrato
tabla_cruzada = pd.crosstab(df['Estrato'], df['Imagen_Rango'], margins=True)
print(tabla_cruzada)


from scipy.stats import chi2_contingency

# aca el csv tiene 9 casos de estrato alto y +20 de los otros(analizable?)
tabla_cont = pd.crosstab(df['Estrato'], df['Imagen_Rango'])
chi2, p, dof, expected = chi2_contingency(tabla_cont)
print("Chi2:", chi2)
print("p-valor:", p)

df['Estrato'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Cantidad de encuestados por Estrato")
plt.ylabel("Cantidad de personas")
plt.show()

df['Imagen del Candidato'].mean

estad_estrato = df.groupby('Estrato')['Imagen del Candidato'].describe()
print(estad_estrato) # NO SIRVE

# Tabla cruzada de conteos
tabla = pd.crosstab(df['Estrato'], df['Imagen_Rango'], margins=True)
print(tabla) # SIRVE 

# Tabla porcentual
tabla_pct = pd.crosstab(df['Estrato'], df['Imagen_Rango'], normalize='index')*100
print(tabla_pct) # SIRVE MAS 


orden = ['Mala', 'Regular', 'Buena', 'Excelente']  # orden de categorías

sns.boxplot(x='Estrato', y='Imagen del Candidato', data=df)
plt.title("Distribución de Imagen del Candidato por Estrato")
plt.show()

#usando la variable fecha
# resample solo funciona si fecha está como índice datetime,
df['Fecha'] = pd.to_datetime(df['Fecha'])
df.set_index('Fecha', inplace=True)
df['Imagen del Candidato'] = pd.to_numeric(df['Imagen del Candidato'], errors='coerce') 
# Función para agrupar por periodo y calcular promedio de imagen y conteo # POSIBLE TRACKING SEMANAL
#AL CSV le falta INTENCION DE VOTO para hacer tracking completo
def tracking_periodo(df, periodo='W'):
    return df.resample(periodo).agg
({'Imagen del Candidato':'mean', 'Estrato':'count'})
tracking_semanal = tracking_periodo(df, periodo='W')
print(tracking_semanal)

tracking_semanal.plot(marker='o')
plt.title("Tracking semanal de imagen e intención de voto")
plt.xlabel("Semana")
plt.ylabel("Promedio")
plt.show()

np.random.seed(42)  # para reproducibilidad
df["intencion_voto"] = np.random.randint(10, 60, size=len(df))  # valores entre 10% y 60%

# Guardar un nuevo CSV con esta variable añadida
nuevo_archivo = r"C:\Users\patri\Downloads\encuesta_candidato_voto.csv"
pd.read_csv(nuevo_archivo)
df.to_csv(nuevo_archivo, index=False)
df = df.dropna() # Eliminar filas con valores NaN

print("Nuevo CSV generado con columna 'intencion_voto'")
print(df.head())
def graficar_tracking(df, variable, candidato):
    subset = df[df['candidato'] == candidato]
    sns.lineplot(x='Fecha', y=variable, data=subset)
    plt.title(f"{variable.capitalize()} - {candidato}")
    plt.show()
    
df['Fecha']
# Agregar una columna "candidato" simulando que hay tres candidatos
np.random.seed(42)
df["candidato"] = np.random.choice(["A", "B", "C"], size=len(df), p=[0.4, 0.35, 0.25])

# Convertir fecha a datetime y establecer como índice
df["Fecha"] = pd.to_datetime(df["Fecha"])
df = df.set_index("Fecha")

print(df.head())
import statsmodels as sm
print("Statsmodels version:", sm.__version__)
X = df[['Edad']]
X = sm.add_constant(X)
y = (df['Imagen del Candidato'] == 1).astype(int)
modelo = sm.Logit(y, X).fit()
print(modelo.summary())

