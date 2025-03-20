import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import os
from PIL import Image
import io

st.title('Dataset original')
st.write("") 
st.write("") 
st.write("") 
st.write("") 


st.subheader('"titanic.csv"')

data = pd.read_csv('data/titanic.csv')
st.dataframe(data)

st.write("") 
st.write("") 
st.write("") 

st.markdown(
    """
    Existen muchas bases de datos que recogen diversa información acerca de los pasajeros del Titanic.
    """)

st.markdown(
    """
    Este dataset importado es una muestra de **891 pasajeros** que se subieron a bordo del Titanic. 
    La cifra exacta de personas de a bordo, contando pasajeros y tripulación, fue de **2225**, 
    de las cuales se salvaron **712** (Wikipedia). En nuestro dataset sólo aparecen pasajeros, 
    aunque, como veremos, algunas de las personas que viajaron en calidad de pasajeros eran 
    en realidad trabajadores, personalidades importantes (como **Thomas Andrews** o **Bruce Ismay**), 
    o empleados personales de éstos (asistentes, secretarios, etc.)
    """)

st.write("") 
st.write("") 
st.write("")

buffer = io.StringIO()
data.info(buf=buffer)
info = buffer.getvalue()
st.text(info)



st.write("") 
st.write("") 
st.write("") 



st.markdown("""## Descriptiva de las variables""")
st.write("") 
st.write("") 
st.write("") 


st.markdown("""Las 12 variables del dataset 'titanic' son las siguientes:

- ***PassengerID***. Cuantitativa discreta. ID del pasajero.
- ***Survived***. Binaria. Supervivencia al naufragio (No = 0; Sí = 1).
- ***Pclass***. Cualitativa ordinal. **integer**. Clase del pasajero (siendo 1 la clase más adinerada y 3 la clase más pobre).
- ***Name***. Cualitativa nominal **string**. Nombre del pasajero.
- ***Sex***. Cualitativa nominal **string**. Sexo ('male' = hombre y 'female' = mujer).
- ***Age***. Cuantitativa discreta **integer**. Edad.
- ***SibSp***. Cuantitativa discreta **integer**. Número de hermanos/as o hermanastros/as del pasajero.
- ***Parch***. Cuantitativa discreta **integer**. Número de padres e hijos en el barco.
- ***Ticket***. Cuantitativa discreta **integer**. ID del billete.
- ***Fare***. Cuantitativa contínua **integer**. Precio (en libras) del billete pagado por el pasajero.
- ***Cabin***. Cuantitativa discreta **integer**. ID del camarote donde se alojó el pasajero.
- ***Embarked***. Cualitativa nominal **integer**. Puerto de embarque del pasajero (S = 'Southampton'; Q = 'Queenstown'; C = 'Cherbourg')""")


st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 

df_cargado = Image.open('img/Titanic_voyage_map.png')

st.image(df_cargado, caption='Itinerario del Titanic. Fuente: Wikimedia Commons', use_container_width=True)
