import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import os
from PIL import Image

st.title('Gráficos')

st.write("") 
st.write("") 
st.write("") 
st.write("") 


# Cargar la imagen guardada
imagen = Image.open(r'..img/sexo_edad_clase.png')


# Mostrar la imagen en la aplicación Streamlit

st.markdown("""## Análisis exploratorio de datos""")

st.markdown("""En este apartado se realizará un análisis exploratorio de las variables del dataset, comenzando por aquellas que, a nuestro juicio son las más importantes. Para las 
variables cuantitativas hallaremos bien el máximo y el mínimo o bien su distribución, mientras que para las discretas nos centraremos en esto último. Dicho análisis estará complementado con
algunas observaciones que nos parezcan dignas de mención.""")

st.markdown("""#### Variables sexo, edad y clase""")



# Tabla de la edad máxima y mínima
df_cargado = pd.read_csv('../tablas/edades.csv')

st.write(df_cargado)



st.image(imagen, caption='Distribución por sexo, edad y clase de pasajero', use_column_width=True)

st.markdown("""Para este análisis, hemos creado una nueva columna que categoriza la variable *Age* utilizando ocho bloques de edad cada 10 años hasta un último bloque que engloba a los mayores de 70
            años. Podemos destacar los siguiente:""")

st.markdown("""
    - Variable *Sexo*: Hay casi el **doble** de hombres que mujeres.
    - Variable *Grupo de Edad*: Destaca sobre el resto el grupo de la **veintena**. Se observa además que hay muchas más personas en los dos primeros grupos (de 0 a 10 años y de 0 a 20) que 
            en los dos últimos. Cabe resaltar que, para este indicador, se han eliminado los valores provenientes de la imputación a aprtir de la mediana de edad.
    - Variable *Clase*: La gran mayoría de personas eran de **tercera clase**, y la clase con menos pasajeros es la segunda.""")

#-----------------------------------------F A R E -------------------------------------------------------------------------
st.write("") 
st.write("") 
st.markdown("""#### Variable Fare""")

st.write("") 


# Tabla de los tres fares más altos de la familia Cardeza
df_cargado = pd.read_csv('../tablas/cardeza.csv')

st.write(df_cargado)
st.markdown("""
    <div style="text-align: center; font-size: 12px; margin-top: 10px;">
        Los tres tickets más caros del Titanic fueron comprados por la familia Cardeza
    </div>
    """, unsafe_allow_html=True)
st.write("")

st.markdown("""Resulta llamativo la gran disparidad de precios que tenían los tickets del Titanic. Pese a encontrarse la gran mayoría de ellos por debajo de las
50 libras, hay unos cuántos que superan la centena e incluso nos encontramos con tres que costaron más de 500 libras.
            
Estos datos están bien recogidos. De acuerdo con la información disponible en la red, Miss Charlotte Wardle Cardeza (ausente en el presente dataset), su hijo Thomas y varios empleados
personales suyos se subieron al Titanic en Cherbourg con un ticket conjunto (lo que explicaría el tan elevado coste) y pudieron sobrevivir al ser rescatados en el tercer
bote salvavidas (Encyclopedia Titanica, 2024)""")

imagen = Image.open('../img/fare1.png')
st.image(imagen, caption='Distribución del fare (sin eliminar outliers)', use_column_width=True)


st.markdown("""Observando la dispersión en el boxplot, parece necesario tomar medidas para deshacernos de los **outliers** en caso de que se quisiese hacer un modelo predictivo. Aunque a continuación
            se muestra el mismo boxplot con los valores que eran atípicos elimiados, de nuevo nos encontramos con algunos outliers. En este caso, la mejor solución puede ser 
            la **transformación logarítmica** de la variable, para mitigar el efecto de estos valores tan altos y no cargarnos información valiosa.
            
Se creará una nueva variable denominada *Log Fare*, que será la que se incluya en el modelo.""")


imagen = Image.open('../img/fare2.png')
st.image(imagen, caption='Distribución del fare (removiendo outliers)', use_column_width=True)


#-----------------------------------------E M B A R K E D -------------------------------------------------------------------------


st.markdown("""#### Variable Embarked""")
st.write("") 
st.markdown("""Algo más del 70% de los pasajeros se embarcaron en Southampton, lugar de partida del trasátlántico. En su primera parada en la ciudad francesa de Cherbourg se subieron a bordo
            el 19 % de ellos, y el 8.64 % restante lo hizo en tierras irlandesas (Queenstown).""")


imagen = Image.open('../img/embarked1.png')
st.image(imagen, caption='Frecuencia de pasajeros según el lugar de embarque', use_column_width=True)


st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 


#-----------------------------------------S I B L I N G S -------------------------------------------------------------------------

st.markdown("""#### Variables SibSp y ParCh""")
st.write("") 
st.write("") 
st.write("") 

st.markdown("""En cuanto a las variables referentes al número de hermanos, padres e hijos, tenemos que:

- Más ó menos una tercera parte de los pasajeros no habían subido con hermanos.
            
- De los 283 que subieron con hermanos, 209 de ellos sólo lo hicieron con uno (representando un 73.9 %)
            
- El 76 % de los pasajeros tenían padre, madre, hijo o hija a bordo.
            
- Entre los que tenían padre/madre o hijo/a, sólo quince viajaban con más de tres de estos familiares""")

st.write("") 
st.write("") 

col1, col2 = st.columns(2)

# En la primera columna (col1)
with col1:
    st.markdown("##### Número de hermanos por pasajero")
    df_hermanos = pd.read_csv('../tablas/hermanos.csv')
    st.write(df_hermanos)

# En la segunda columna (col2)
with col2:
    st.markdown("##### Padres o hijos por pasajero")
    df_padres_hijos = pd.read_csv('../tablas/padres_hijos.csv')
    st.write(df_padres_hijos)
         
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 


#-----------------------------------------C O R R E L A C I O N E S -------------------------------------------------------------------------


st.markdown("""## Matriz de correlaciones""")

imagen = Image.open('../img/correlaciones.png')
st.image(imagen, caption='Matriz de correlaciones de las variables', use_column_width=True)

st.markdown("""Destacamos las siguientes relaciones:
    
    - Una fuerte relación inversamente proporcional entre el Fare y la clase.
    
    - Una fuerte relación entre el sexo y la supervivencia.
    
    - Una fuerte relación entre tener o no una cabina conocida y el Fare.
            
    - Moderada relación entre la clase y la probabilidad de sobrevivir.
    
    - Moderada relación entre tener una cabina conocida y sobrevivir.
            
    - Moderada relación entre la edad y la clase (a mayor edad, menor clase).""")

st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 



st.markdown("""## El enigma de los tickets""")

st.write("") 
st.write("") 
st.write("") 


st.markdown("""### ¿Por qué algunos tickets valían cero?""")

st.write("") 
st.write("") 


st.markdown("""- Tickets LINE: Explicados en el apartado *limpieza datos*.
""")



st.markdown("""- **Ticket de Thomas Andrews**, el ingeniero responsable de la construcción del Titanic (ticket *112050*). Pasaje gratis por ser el encargado de la construcción.
            
- Tickets que empiezan por '239...':

El conjunto de tickets que empiezan por '239...' y que figuran con un *Fare* igual a 0 pertenecen todos a un grupo de empleados llamado *Guarantee Group* (**Grupo de Garantía**). 
Esta gente había trabajado en la construcción del barco y habían sido invitados por Thomas Andrews para este viaje inaugural como honra a su destacado trabajo. 
Se les permitió un pasaje gratuíto (Titanic Talks, 2020).

No obstante, otras fuentes, como el documental *Titanic: Birth of a Legend* (2005), apuntan a que, en realidad, el Guarantee Group tenía como cometido la supervisión de los barcos en cada viaje inaugural, para detectar 
problemas, típicos en los viajes inaugurales y ofrecer soluciones, información que también corrobora la Encyclopedia Titanica.""")  

df_cargado = Image.open('../img/tique_real.jpg')

st.image(df_cargado, caption='Un ticket de tercera clase del Titanic. Fuente: Flickr', use_column_width=True)

st.markdown("""- Tickets que empiezan por '112...':

Tanto los señores William Harrison como Richard Fry eran, respectivamente, el secretario y el asistente personal del mismísimo **Joseph Bruce Ismay** (ausente en esta lista),
el director de la compañía constructora del Titanic White Star Line. Como es lógico, sus tickets eran gratuítos. 
El caso de **William Parr** es más complejo: se trataba de un ingeniero eléctrico que también formaba parte del Guarantee Group pero, 
a diferencia de sus compañeros, viajaba en primera clase (Encyclopedia Titanica, 2024).


- Ticket número 19972:

El caso de **John George Reulchin** es algo más curioso. Se trataba de un empleado de la compañía **Holland America Line**, que estaba en la misma matriz empresarial que la White Star Line, 
por lo que su pasaje le salió gratuito. Como los miembros del Guarantee Group, también tenía trabajo durante este viaje inaugural, 
ya que desde Holland American Line le habían enconmendado evaluar a los trasatlánticos de clase olímpica porque iban a empezar a fabricarlos también (Encyclopedia Titanica, 2024). 
""")

st.write("") 
st.write("") 
st.write("") 
st.write("") 


st.markdown("""### ¿Qué proporcion de tickets llevaba prefijo?""")

st.write("") 
st.write("") 

st.markdown("""Pero lo que sin duda llama más la atención de los tickets es el **prefijo** que llevaban algunos de ellos. Representaban casi un **26% sobre el total** de 
            tickets (véase en la siguiente gráfica de tarta) y, además, había muchos tipos de prefijos diferentes.""")


df_cargado = Image.open('../img/ratio_letras.png')

st.write(df_cargado)

st.markdown("""**¿Qué prefijos eran los más habituales? ¿Es lógico agruparlos?**
            
            
Pues parece que sí. Si le echamos una ojeada a la columna *Tickets*, observamos varios tipos de prefijos que se repiten periódicamente. Algunas veces observamos un mismo prefijo escrito 
de forma ligeramente diferente (véanse los prefijos *CA* y *C.A.*). Otras veces, lo que sucede es que hay dos tipos de prefijo muy similares, como los 
prefijos *A/4* y *A/5*, o como los *STON* y los *SOTON*.
            
Tras estas observaciones, hemos decidido agrupar, de manera un tanto artificial, a los tickets con prefijos en varios grupos: 
            
- Los que tienen el prefijo *PC*
            
- Los que tienen el prefijo *CA* o *C.A.*
            
- Los que tiene el prefijo *A/4* o *A/5*

- Los que tiene el prefijo *STON o SOTON*""")

st.write("") 
st.markdown("""En la siguiente gráfica se puede comprobar la frecuencia de cada tipo de prefijo dentro de los tickets que llevaban prefijo. Como vemos, casi **tres cuartas partes** de ellos
están en alguno de los grupo que hemos creado""")

st.write("") 
st.write("")


df_cargado = Image.open('../img/ratio_prefijos.png')

st.write(df_cargado)
st.write("") 
st.write("")

st.markdown("""### ¿Qué significado podrían tener estos prefijos? ¿Es posible averiguarlo?""")
st.write("") 
st.write("")
# PC

st.markdown("""##### Billetes *PC*""")

st.markdown("""A pesar de que en algunos foros de la Encyclopedia Titanica se ha escrito que *PC* podría significar *Private Cabin* o *Private Class*, no parece muy lógico que 
sea este el significado, pues muchos de los pasajeros de primera clase no poseían estos tickets y tenían cabina privada.
            
Lo que sí podría tener más sentido es que estos prefijos estuvieran relacionados con la **agencia de viajes** que los vendía, y que esa agencia llevase en su nombre las letras P y C.
            
Si nos fijamos, todos estos tickets son de Clase 1 y la mayoría embarcaron en **Cherbourg**. Esto apunta a que la compra internacional de tickets del Titanic se hiciese,
entre otras, a través de esta supuesta compañía, y por personas de muy altopoder adquisitivo.""")
df_cargado = pd.read_csv('../tablas/PC.csv')
st.write(df_cargado)

df_cargado = Image.open('../img/pc_cherbourg.png')

st.write(df_cargado)

st.write("") 
st.write("") 
st.write("") 
st.write("") 


# A/5


st.markdown("""##### Billetes 'A/5 o A/4'""")

st.markdown("""Todos los tickets de este tipo son de **tercera clase**, pero llama la atención que, de los 23 que hemos encontrado, 21 de ellos pertenezcan a hombres y sólo dos de ellos a mujeres. 
Teniendo en cuenta que entre los pasajeros de tercera clase hay unos 2.5 hombres por cada mujer, en este grupo de billetes hay una clara **infrarrepresentación** de mujeres.
            
¿A qué podría ser esto debido? ¿Simple estadística? O quizá no... ¿Podría ser que estos tickets se vendiesen en algún lugar o contexto en donde sólo había hombres?
            
De nuevo, la duda queda en el aire.""")

df_cargado = pd.read_csv('../tablas/A5.csv')

st.write(df_cargado)

st.write("") 
st.write("") 
st.write("") 
st.write("") 


# CA


st.markdown("""##### Billetes 'C.A. o CA'""")

st.markdown("""Esta clase de billetes se vendieron a pasajeros de segunda y de tercera clase, prácticamente en la misma proporción. Todos ellos embarcaron en Southampton. 
Apenas vemos datos relevantes, salvo una curiosidad: la existencia de un **ticket** que incluye los prefijos C.A. y SOTON.
            
¿Como es esto posible?""")


df_cargado = pd.read_csv('../tablas/CA.csv')

st.write(df_cargado)

st.write("") 
st.write("") 
st.write("") 
st.write("") 



# STON


st.markdown("""##### Billetes 'STON o SOTON'""")

st.markdown("""Veamos el último de los grupos:
            
Se vendieron un total de 36 tickets de este tipo, **todos** ellos de **tercera clase**. Únicamente hubo 6 mujeres que lo adquirieron, por lo que también aquí hay una infrarrepresentación del sexo femenino.
No obstante, en cuanto al significado del prefijo, sí que podemos deducir que hace alusión a la ciudad de **Southampton**.
            
- *SOTON* es un diminutivo de Southampton, y es probable que hace 100 años también se utilizase la forma abreviada *STON* para referirse a esta ciudad.
Cabe decir que, a día de hoy, también se utiliza la forma *So'ton* (Wikictionary, 2024).
- Todos estos tickets pertenecían a pasajeros que se subioeron en esta ciudad.
            
Parece claro que el target de estos tickets eran personas de clase media baja que estaban trabajando en Southampton y alrededores, y que posiblemente buscaban
retornar a América (si eran originarios de allí) o irse a América a buscar una nueva vida. En cualquier caso, no tenemos información acerca de
la compañía que vendía este tipo de tickets.""")



st.markdown(""" """)

df_cargado = pd.read_csv('../tablas/STON.csv')

st.write(df_cargado)
            
            
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 


st.markdown("""## 6. Análisis de supervivencia""") 

st.write("") 
st.write("") 
st.write("") 
st.write("") 


st.markdown("""### Supervivencia por Sexo""")

st.markdown("""Sobrevivieron un 74 % de mujeres y un 19 % de hombres.""")

data = {
    'Sexo': ['Mujeres', 'Hombres'],
    'Tasa de Supervivencia': [0.7420, 0.1889]
}

df = pd.DataFrame(data)

# Cambiar los estilos de la tabla
st.markdown(
    """
    <style>
    .styled-table {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 22px;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    .styled-table th, .styled-table td {
        padding: 12px 15px;
        text-align: center;
    }
    .styled-table th {
        background-color: #4CAF50;
        color: white;
    }
    .styled-table tr {
        border-bottom: 1px solid #77dddd;
    }
    .styled-table tr:hover {
        background-color: #666666;
    }
    </style>
    """, unsafe_allow_html=True
)

# Mostrar la tabla con estilo
st.markdown('<div class="styled-table">', unsafe_allow_html=True)
st.table(df)
st.markdown('</div>', unsafe_allow_html=True)

st.write("") 
st.write("") 
st.write("") 
st.write("") 
st.write("") 


st.markdown("""### Supervivencia por grupos de tickets""")

st.write("") 
st.write("") 


df_cargado = Image.open('../img/supervivencia_tickets.png')

st.write(df_cargado)

st.write("") 
st.write("") 
st.markdown("""Observamos que las personas con tickets *PC* son las que **más sobrevivieron**, lo cual concuerda porque se trata sólo de pasajeros
de primera clase. El lado opuesto son las personas con tickets con prefijo del tipo *A/5*, de los 
cuales **sólo 2** fueron afortunados.
            
En los billetes tipo *CA* y *STON la supervivencia es intermedia, pero los de tipo *CA* favorecen más al sexo famenino.
            
Estos resultados se entienden mejor cuando analizamos la supervivencia por clase y sexo. Como vemos en la gráfica de *Supervivencia por clase*, las mujeres de clase 2
tenían más del 90% de probabilidades de sobrevivir, y el hecho de que en los tickets *CA* **haya personas de segunda** y tercera clase, 
en contraposición a los que ocurre con los tickets *STON* o los *A/5* puede explicar esa diferencia de supervivencia entre hombres y mujeres""")
st.write("") 
st.write("") 

st.markdown("""### Supervivencia por Clase""")

df_cargado = Image.open('../img/supervivencia_clase.png')

st.write(df_cargado)

st.write("") 
st.write("") 
st.write("") 
st.write("") 

st.markdown("""### Supervivencia por Grupo de Edad""")
df_cargado = Image.open('../img/supervivencia_edad.png')

st.write("") 
st.write("") 
st.markdown("""Finalmente, toca repasar la supervivencia para cada franja etaria. Se observa que, entre el grupo **menor de 10 años**, tanto en hombres como mujeres la tasa de supervivencia 
es del 0.6, pero inmediatamente en el grupo de 10 a 20 años la tasa en hombres se desploma, manteniéndose **entre el 0.2 y el 0.1** hasta la última franja.
            
Por en contrario, en mujeres parece que hay un **aumento** de la tasa **con la edad**, llegando incluso al 100% para las mujeres de más de 60 años.""")

st.write(df_cargado)
