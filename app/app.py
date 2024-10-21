import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import os
from PIL import Image


# Inyectar CSS para Google Fonts directamente en el encabezado de la app
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Bodoni:wght@700&display=swap');

    .title-text {
        font-family: 'Libre Bodoni', serif;
        font-size: 50px;
        color: #772211;
        background-color: #f5f5dc;
        text-align: center;
        border-radius: 10px;
    }

    </style>
    """, unsafe_allow_html=True)
st.markdown('<h1 class="title-text">T I T A N I C</h1>', unsafe_allow_html=True)

st.title('**Lo que aún no sabemos...**')



# Tu nombre
st.write('**Presentado por: Estevo Arias García**')  # Reemplaza [Tu Nombre] con tu nombre real

# Barra lateral
st.sidebar.title('Menú de la Aplicación')

# Cargar imagen
imagen = Image.open('../../img/titanic_belf.jpg')
st.image(imagen, caption='Titanic saliendo de Queenstown el 11 de abril de 1912. Fuente: Wikimedia Commons', use_column_width=True)

# Espacio adicional para un diseño limpio
st.write('---')  # Línea horizontal para separar contenido
st.write("""##### En este trabajo revelaremos datos sorprendentes sobre el Titanic""")


st.write("") 
st.write("") 

st.markdown("""## Descubriremos...""")
st.markdown("""##### - Datos reales de algunos pasajeros que no aparecen.""")
st.markdown("""##### - Qué se esconde tras los pasajeros con tickets 'LINE'""")
st.markdown("""##### - Qué era el Guarantee Group de Thomas Andrews. """)

st.write("") 
st.write("") 
st.write("") 


st.markdown("""## Analizaremos...""")
st.markdown("""##### - Los curiosos prefijos de los tickets.""")
st.markdown("""##### - La supervivencia por edad, sexo y clase.""")
st.markdown("""##### - Crearemos un modelo predictivo.""")

st.write("") 
st.write("") 
st.write("") 

st.markdown("""## Referencias""")


st.markdown("""- 1) Encyclopedia Titanica, (2024). William Henry Tornquist Disponible en: https://www.encyclopedia-titanica.org/titanic-survivor/william-henry-tornquist.html
- 2) Blog | Titanic Talks, (2024): Disponible en: https://milliehaworth.wixsite.com/website/post/titanic-connections-titanic-s-guarantee-group
- 3) Wikipedia, (2024). Pasajeros a bordo del RMS Titanic. Disponible en: https://es.wikipedia.org/wiki/Anexo:Pasajeros_a_bordo_del_RMS_Titanic
- 4) Encyclopedia Titanica, (2024). Disponible en: https://www.encyclopedia-titanica.org/
- 5) Bank of England. Inflation Calculator, (2024). Disponible en: https://www.bankofengland.co.uk/monetary-policy/inflation/inflation-calculator
- 6) Encyclopedia Titanica, (2024). Charlotte Wardle Cardeza. Disponible en: https://www.encyclopedia-titanica.org/titanic-survivor/charlotte-cardeza.html
- 7) Titanic. Birth of a Legend, (2024). The Guarantee group. Disponible en: https://atyeo.co.uk/titanic/pages/men-2.html
- 8) So'ton. Wikictionary. Disponible en: https://en.wiktionary.org/wiki/So%27ton#English""")
