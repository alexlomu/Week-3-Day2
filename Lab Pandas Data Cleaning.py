import pandas as pd
import numpy as np

data = pd.read_csv("Vehicles_messy_clean.csv")
# Inspeccionamos las 5 primeras filas del csv
print(data.head())
# Miramos el tamaño del archivo
print(data.shape)
# Concluimos que es 5x15
# Nombres de columnas, el tipo de data y los valores nulos
data.info(memory_usage="deep")
# Numero de valores nulos solo de las columnas que tengan estos valores
data_nulo = pd.isnull(data).sum()
data_nulo[data_nulo >0]
# Eliminamos las columnas que tengan mas de 10000 valores nulos
columnas_nulas = data_nulo[data_nulo>1e4].index
print(len(columnas_nulas))
data.drop(columnas_nulas, axis = 1, inplace = True)
print(len(data.columns.values))
# Creamos un nuevo dataframe que muestre solo las filas en las que displ es nulo
data_displ_nulo = data[data.displ.isnull() == True]
# Rellenamos los valores que sean nan 
# Para ello primero miramos que tenemos que rellenar
data[["cylinders", "displ", "drive", "trany"]].info()
data.cylinders.fillna(0, inplace = True)
data.displ.fillna(0, inplace= True)
# Bonus
data.drive.fillna("unknown", inplace = True)
data.trany.fillna("Automatico", inplace = True)
# Incorrect values
incorrectos = data[((data.cylinders==0)&(data.displ!=0))|(data.cylinders!=0)&(data.displ==0)].index
data.loc[incorrectos,["make","model","fuelType","cylinders","displ","drive","trany","year"]]
# Hay columnas con valores ecxtraños?
data.loc[incorrectos, "cylinders"] = 2.0
print(data.loc[incorrectos, "cylinders"])
# Check similar rows
data.loc[incorrectos, "cylinders"] = 4.0
print(data.loc[incorrectos, "cylinders"])
# Bonus

# Low Variance Columns
variedad = []
for row in data.select_dtypes(include =np.number):
    percentil90 = np.percentile(data[row],90)
    if percentil90==data[row].min():
        variedad.append(row)
print(len(variedad))
# Drop method 
data.drop(variedad, axis = 1, inplace= True)
# Extreme Values and Outliers

centro = data.describe().transpose()
centro["IQR"] = centro["75%"]-centro["25%"]
print(centro)

extremos = pd.DataFrame(columns = data.columns)

for col in centro.index:
    iqr = centro.at[col, "IQR"]
    recorte = iqr * 1.5
    superior = centro.at[col,"25%"] - recorte
    inferior = centro.at[col,"75%"] + recorte
    extrem = data[(data[col] < inferior) | (data[col] > superior)].copy()
    extrem["extremos"]= col
    extremos = extremos.append(extrem)
# Usamos head para comprobar que todo ha salido bien
extremos.head()

# Data Type Correction

# Check the data type of each column
data.info(memory_usage="deep")

# Cleaning years
data.year = pd.to_datetime(data["year"], format="%Y")
# Ahora vamos a cambiar el tipo de dato de todo lo que nos interesa
for row in data.select_dtypes("object").columns:
    data[row] = data[row].astype("category")
# Comprobamos que los cambios se han hecho correctamente
data.info(memory_usage="deep")

# Cleaning Text and Removing Special Characters

# Check the unique values
print(data.trany.unique())
#Use replace
data.trany = data.trany.str.replace(" ","")
data.trany = data.trany.str.replace("-","")
data.trany = data.trany.str.replace("(","")
data.trany = data.trany.str.replace(")","")
# Comprobamos que rodo ha cambiado
print(data.trany.unique())

# Finding and Removing Duplicates

# Drop duplicate rows that are completely equal
# Comprobamos si hay duplicados en el dataframe
print(len(data.index) == len(data.drop_duplicates().index))
# Select a subset of columns, remove all other columns, and then use the drop_duplicates method to drop any duplicate records based on the remaining columns.
subsetData = data [["fuelType","make","model"]]
subsetData = subsetData.drop_duplicates()

# Export clean dataset
data.to_csv("csv_limpio.csv",index = False)
