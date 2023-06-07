
#IMPORTACION DE LIBRERIAS

#Importamos pandas para el manejo de los datos
import pandas as pd
import numpy as np

#Importamos matplotlib para la visualizacion de los datos
import matplotlib.pyplot as plt

#Importamos las librerias de sklearn para desarrollar los modelos de Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE 
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate
import sklearn.metrics as metrics

#Importamos los warnings del modelo y los ignoramos para visualizar correctamente los datos dentro del IDE
import warnings
warnings.filterwarnings("ignore")

#Realizamos la importacion de los datos en el archivo .xlsx y almacenamos dentro de un DataFrame; Renpombramos 'fecha' por 'date'
df = pd.read_excel('Prueba - Data Scientist.xlsx', parse_dates=['fecha']).rename(columns={"fecha":"date"})

#ELIMINACION COLUMNA FAMILIA
df.drop("familia", axis=1, inplace=True)
df

#Graficamos las ventas en el tiempo para cada producto
count = 0
for t in df['codigo'].unique():
  
  dfr = df[df['codigo']==t]
  plt.plot('date', 'unidades_venta' ,data=dfr, label=t, linestyle=':' )
  plt.legend(loc='upper left')
  plt.show()
 #SI DESEA DESCARGAR LAS GRAFICAS EN FORMATO .jpeg QUITAR COMENTARIO DE LAS SIGUIENTES DOS (2) LINEAS (Estas se guardaran en la carpeta de ejecución)
  #count = count+1 
  #plt.savefig("GraficaProd{:03}.jpeg".format(count))

#Solicitamos la descripcion de los datos para ver un resumen de las ventas por producto.
df.groupby("codigo").describe()

#Creamos un DataFrame vacio donde guardaremos los forecast de cada producto para los siguientes 6 meses
dforecast = pd.DataFrame()
#Creamos un DataFrame con la informacion de cada producto 
for t in df['codigo'].unique():
    dfPrueba = df[df['codigo']==t]
  #Extraemos los meses y años de la fecha de cada producto
    dfPrueba['Month'] = [i.month for i in dfPrueba['date']]
    dfPrueba['Year'] = [i.year for i in dfPrueba['date']]
  #Creamos una secuencia de numeros 
    dfPrueba['Series'] = np.arange(1,len(dfPrueba)+1)
  #Eliminamos la columna 'date'
    dfPrueba.drop(['date'], axis=1, inplace = True)
  #Reconstruimos el DataFrame con la secuencia de numeros, año, mes y unidades vendidas. 
    dfPrueba = dfPrueba[['Series', 'Year', 'Month', 'unidades_venta']]
  #Dividimos los datos para el eje X y eje Y
    feature_cols= ['Year','Month']
    X= dfPrueba[feature_cols]
    Y= dfPrueba['unidades_venta']
  #Establecemos las variables de entrenamiento y de prueba para el modelo de Machine Learning 
    trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.16, random_state=0)
  #Importamos la clase de Regresion Lineal, la instanciamos y llamamos al metodo fit con parametros los datos a entrenar
    lm = LinearRegression()
    lm.fit(trainX,trainY) 
  #Creamos un estimador y selector que mejore el modelo de Machine Learning 
    estimador= SVR(kernel="linear")
    selector = RFE(estimador,n_features_to_select=1,step=1)
    selector = selector.fit(X,Y) 
  #Imprimimos el Intercepto, Coeficiente y el R^2
    print(f'{t}: Intercepto: {lm.intercept_} Coeficientes [año mes]: {lm.coef_} R2: {lm.score(trainX,trainY)}')
  #Comparamos la salida actual para los valores de prueba'testX' con los valores pronosticados
    y_pred = lm.predict(testX)
    #dfPrueba = pd.DataFrame({'Actual': testY.to_numpy().flatten(), 'Pronosticado': y_pred.flatten()})
    #dfPrueba = pd.DataFrame({'codigo': t, 'Pronosticado': y_pred.flatten()})
    #LLenamos el dataframe con los datos deseados para cada producto
    dforecast = dforecast.append({
        'codigo': t, 
        'forecast 1': y_pred.flatten()[0], 
        'forecast 2': y_pred.flatten()[1],
        'forecast 3': y_pred.flatten()[2],
        'forecast 4': y_pred.flatten()[3],
        'forecast 5': y_pred.flatten()[4],
        'forecast 6': y_pred.flatten()[5] },
         ignore_index=True)

    #print('_______DataFrame de Prueba_______')
    #print(dfPrueba['Pronosticado'])
    #print(dfPrueba)
    #print('_______DataFrame de Prueba------------')
    
    #df1 = dfPrueba.head(36)
    #df1.plot(kind='bar', figsize=(36,20),label=t)
    #dfPrueba.plot(kind='bar', figsize=(36,20),label=t)
    #plt.grid(which='major',linestyle='-', linewidth='0.5',color='green')
    #plt.legend(loc='upper left')
  
    #print(trainY.to_numpy())
    #plt.scatter(testX.to_numpy().flatten(),testY.to_numpy().flatten(), color='gray')
    #plt.plot(testX.to_numpy(), y_pred.flatten(), color='red', linewidth=2)
    #plt.show()
    print('Mean Absolute Error:', metrics.mean_absolute_error(testY, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(testY, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(testY, y_pred)))

    writer = pd.ExcelWriter('Forecast.xlsx')
    dforecast.to_excel(writer, sheet_name='Punto 1')
    writer.save()

"""Ejemplo - limpieza de datos

Organizamos la informacion de los productos en una tabla que los agrupe por codigo y fecha para ver el historial de ambos juntos y asi encontrar una relacion entre las ventas del producto y su existencia en inventario
"""

#-------------------------------Convertimos los datos del excel en dataframe para empezar con su limpieza
df2ventas = pd.read_excel('Prueba - Data Scientist.xlsx', parse_dates=['fecha'], sheet_name=1)
df2inventario = pd.read_excel('Prueba - Data Scientist.xlsx', parse_dates=['fecha'], sheet_name=2)

#-------------------------------Como primera medida vamos a revisar que no haya datos invalidos en ambas tablas
df2ventas.notnull()
df2ventas.dropna()

df2inventario.notnull()
df2inventario.dropna()

#df2ventas.describe()
#df2inventario.describe()

#-------------------------------revisaremos los datos cruzando la informacion de ventas con inventario

alldata = df2ventas.append(df2inventario)

#Recorremos la tabla agrupada y examinaremos que el inventario del dia sea igual a la resta del inventario anterior y las ventas de hoy
agrupado = alldata.groupby(['codigo','fecha']).mean()

agrupado.head(50)

writer = pd.ExcelWriter('depurados.xlsx')
agrupado.to_excel(writer, sheet_name='Punto 2')
writer.save()
