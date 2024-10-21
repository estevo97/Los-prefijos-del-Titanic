import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import os
import joblib
from PIL import Image

st.title('Modelo predictivo')

model_path = 'model\model_XGB.pkl'

# Cargar el modelo
model = joblib.load('model\model_XGB.pkl')

st.write("Modelo de regresión logística para predecir la supervivencia")

st.write("### Codificación del dataset y cambios en las columnas")

st.markdown("""Para poder implementar este modelo predictivo:
            
- Se ha creado una nueva columna con la **transformación logarítmica** de los datos de *Fare* con el fin de mitigar el efecto de los outliers.
- Se ha cerado una nueva columna llamada *Tipo_cabina*, que es **binaria**. Indica si una cabina de un pasajero es conocida o es desconocida.
- Se ha creado una nueva Columna llamada *Prefijo_ticket* que indica el grupo de ticket que tiene el pasajero según los **prefijos** que hemos comentado o,
en caso de que no pertenezca a ninguno, se indica que no tiene prefijo.
            
Se han probado tres tipos de modelos: uno logístico otro KNN y otro Random Forest. Se escogió el Modelo logístico por obtener mejor ratio de acierto (0.8516).""")


st.write("### Predicción de Supervivencia en el Titanic")

# Descripción para el usuario
st.write("""
Introduce las siguientes características del pasajero para predecir si habría sobrevivido en el Titanic:
""")


# Inputs para que el usuario proporcione los datos
Pclass = st.selectbox('Clase del ticket (Pclass)', [1, 2, 3], index=2)
Sex = st.selectbox('Sexo (0 = Female, 1 = Male)', [0, 1], index=1)
Age = st.number_input('Edad', min_value=0, max_value=100, value=25)
SibSp = st.number_input('Número de hermanos/esposos abordo (SibSp)', min_value=0, max_value=8, value=0)
Parch = st.number_input('Número de padres/hijos abordo (Parch)', min_value=0, max_value=6, value=0)
Embarked = st.selectbox('Puerto de embarque (Embarked: 0=Cherbourg, 1=Queenstown, 2=Southampton)', [0, 1, 2], index=2)
Tipo_cabina = st.selectbox('Tipo_cabina (0 = Desconocida, 1 = Conocida)', [0, 1], index = 0)
Prefijo_ticket = st.selectbox('Prefijo_ticket (0 = A/5, 1 = CA, 2 = PC, 3 = STON, 4 = Sin Prefijo)', [0, 1, 2, 3, 4], index = 4)
Fare = st.number_input('Tarifa (Fare)', min_value=0.0, value=7.25)


# Aplicar transformación logarítmica a Fare
LogFare = np.log(Fare) if Fare > 0 else 0  # Evitar -inf si Fare es 0

# Crear un array con los datos de entrada del usuario (usando LogFare)
input_data = np.array([[Pclass, Sex, Age, SibSp, Parch, Embarked, Tipo_cabina, Prefijo_ticket, LogFare]])

# Botón para hacer la predicción
if st.button('Predecir'):
    # Hacer la predicción usando el modelo cargado
    prediction = model.predict(input_data)
    
    # Mostrar el resultado
    if prediction[0] == 1:
        st.write('El modelo predice que el pasajero habría **sobrevivido**.')
    else:
        st.write('El modelo predice que el pasajero **no habría sobrevivido**.')
