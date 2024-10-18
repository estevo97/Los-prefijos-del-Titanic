import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import os
import io
from PIL import Image

st.title('Limpieza de datos')

data = pd.read_csv('../data/titanic.csv')






# Mostrar la estructura del DataFrame (info equivalente)
st.write("### Información sobre el DataFrame:")

# Usar StringIO para capturar la salida de data.info()
buffer = io.StringIO()
data.info(buf=buffer)
info = buffer.getvalue()

# Mostrar el contenido de data.info() en Streamlit
st.text(info)



st.write("") 
st.write("") 
st.write("") 
st.markdown("""Tres son las variables que contienen valores nulos: *Age* (edad), *Cabin* (camarote) y *Embarked* (lugar de embarque). Sólo la edad y el camarote contienen un 
            porcentaje de nulos destacable. La variable *Cabin* va a ser imposible de imputar, pero, para la implementación del modelo predictivo, crearemos una nueva variable a 
            partir de esta que sea binaria, es decir: pasajeros con camarote conocido o con camarote desconocido.""")
st.write("") 
st.write("") 



data = pd.read_csv('../mi_dataframe.csv')
st.write("### Porcentaje de nulos por columna")
st.dataframe(data)

st.write("") 
st.write("") 
st.write("") 
st.write("") 

st.write("### Imputación de la *Edad*:")

st.markdown("""Al tratarse de una variable cuantitativa discreta, se imputarán los valores a partir de la mediana de la misma. La mediana de edad de los pasajeros de este dataset
es de 28.0 años, por lo que se sustituirán los valores faltantes por esta cantidad.""")

st.write("") 
st.write("") 
st.write("") 

st.write("##### Imputaciones especiales")


st.markdown("""El número total de nulos para la columna *Embarked* es de 2. Siendo así, podemos **inferir** directamente los datos de *Embarked* **consultando las fuentes**, pues es sabido que en internet hay una gran cantidad de información acerca 
            de este suceso. Uno de los sitios más recomendados para esta labor es *Encyclopedia Titanica*, una web que posee información completísima sobre el Titanic. 
            De esta forma, descubrimos que las pasajeras Miss. Amelie Icard y Mrs. George Nelson Stone (Martha Evelyn) se subieron a bordo del Titanic en Southampton.""")
st.write("") 

st.markdown("""Aunque en la columna *Fare* no hay nulos, llama la atención la gran cantidad de ceros. Esto puede ser debido a que realmente esos pasajeros (cuyo *Fare* sea igual 
a cero), no pagaron nada para subir al Titanic, o bien, que haya datos mal recogidos. Hciendo un simple 'data[data['Fare]==0]' hallamos 15 ceros. Cuatro de esos valores llaman la atención por el prefijo 'LINE' que llevan 
en el ticket. Investigando, descubrimos la verdadera cuantía de estos, 7.5 libras.""")


st.write("")
         

st.markdown("""- Tickets **LINE**: El Fare real es de **7.5 libras**.

De acuerdo con *Encyclopedia Titanica*, cinco empleados de la compañía naviera *American Line* habían sido acogidos como pasajeros de tercera clase en el Titanic al módico (entre comillas)
precio de 7.5 libras de la época (unas 700 libras actuales de acuerdo con el Bank of England). El caso más conocido fue el del único superviviente, William Henry Tornquist, pero hubieron otros cinco más: Andrew John Shannon (Mr. Lionel Leonard en la lista), 
Alfred Johnson, William Cahoone Johnson y otros dos que no aparecen en esta lista. 

La acogida se debió a la cancelación del Philadelphia, que les habría retornado a los Estados Unidos de no ser por las huelgas de los mineros ingleses que privaban de carbón a trenes y trasatlánticos y, por otra parte, al hecho de que el poco carbón disponible iba a ser utilizado para la gran inauguración del Titanic.
""")

imagen = Image.open(r'../img/huelga_carbon.jpg')
st.image(imagen, caption='Huelga de carbón de mineros ingleses. Fuente: https://boudewijnhuijgens.getarchive.net/', use_column_width=True)


