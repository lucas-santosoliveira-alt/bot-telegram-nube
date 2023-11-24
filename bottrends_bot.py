# Importamos las librerías
from config import *                           # Importamos el token
import telebot                                 # librería de la API de Telegram
from telebot.types import ReplyKeyboardMarkup  # Acceso a usar botones
from telebot.types import ForceReply           # Acceso a citar mensajes
from telebot.types import ReplyKeyboardRemove  # Acceso borrar botones
import time

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import json
import re
import os
from datetime import datetime, timedelta

#archivos 
archivos = ["city", "category_google", "category_yelp", "business_google", "business_yelp", "busimess_category_google",
            "business_category_yelp", "reviews_google", "reviews_yelp", "business_horarios_google", "misc_google", "business_misc_google", 
            "attribute_yelp", "business_attribute_yelp"]

'''
from google.cloud import storage

# Cargamos los dataframes que necesitamos con los datos de los parquet que están google cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gothic-sled-403017-44ae22019b45.json"
storage_client = storage.Client()

my_bucket = storage_client.get_bucket("empresas-yelp")

#  Funcion para descargar archivos
def download_file_from_bucket(blob_name, file_path, bucket_name):
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        with open(file_path, "wb") as f:
            storage_client.download_blob_to_file(blob, f)
    except Exception as e:
        print(e)
        return False

# Carga de archivos
bucket_name = "empresas-yelp"
archivos = ["city"]
df = {}
for archivo in archivos:
    download_file_from_bucket(f'empresas_yelp/{archivo}.parquet', os.path.join(os.getcwd(), 'data'), bucket_name)
    df[archivo] = pd.read_parquet('data')
'''

########
df = {}
for archivo in archivos:
    archi = f'parquet/{archivo}.parquet'
    df[archivo] = pd.read_parquet(archi)
#######

usuarios = {} # Creamos el diccionario para guardar solicitudes del usuario
df_similares = {} # Creamos el diccionario para almacenar los dataframes que se usaran en los modemos de recomendación

########

# ****************************************************************************************************************************** #
# *** Creación y entrenamiento del modelo de MACHINE LEARNING para recomendaciones basado en la similitud del coseno         *** #
# ****************************************************************************************************************************** #

def aplano_listas(conjunto, datataframe, nombre_campo, nombre_registro, lista_campos):
    archivo = datataframe[datataframe["business_id"].isin(conjunto)].sort_values(by='business_id', ascending=True)
    archivo.reset_index(drop=True, inplace=True)

    id = ""
    registros = []
    lista = []
    for index, row in archivo.iterrows():
        if row["business_id"] != id:
            if index != 0:
                lista.append({"business_id": id, nombre_campo: ",".join([valor for valor in registros])})
                registros = []

            id = row["business_id"]
        registros.append(",".join([row[valor] for valor in lista_campos]))
    
    if registros:  # Manejar el último registro si no está vacío
        lista.append({"business_id": id, nombre_campo: ",".join([valor for valor in registros])})

    return pd.DataFrame(lista)

def aplano_listas_google(conjunto, datataframe, nombre_campo, nombre_registro, lista_campos):
    archivo = datataframe[datataframe["gmap_id"].isin(conjunto)].sort_values(by='gmap_id', ascending=True)
    archivo.reset_index(drop=True, inplace=True)

    id = ""
    registros = []
    lista = []
    for index, row in archivo.iterrows():
        if row["gmap_id"] != id:
            if index != 0:
                lista.append({"gmap_id": id, nombre_campo: ",".join([valor for valor in registros])})
                registros = []

            id = row["gmap_id"]
        registros.append(",".join([row[valor] for valor in lista_campos]))
    
    if registros:  # Manejar el último registro si no está vacío
        lista.append({"gmap_id": id, nombre_campo: ",".join([valor for valor in registros])})

    return pd.DataFrame(lista)

def genera_business_similar():
    ####### YELP
    df_similares["master"] = df["business_yelp"]
    yelp_set = set(df_similares["master"]["business_id"])
    inyectado = df["business_category_yelp"].merge(df["category_yelp"], on="category_id")
    df_similares["master"] = df_similares["master"].merge(aplano_listas(yelp_set, inyectado, "categoy", "category", ["category"]), on="business_id", how='left')
    inyectado = df["business_attribute_yelp"].merge(df["attribute_yelp"], on="attribute_id")
    inyectado["attributes"] = inyectado["attribute"] + ">>" + inyectado["value"]
    inyectado["attributes"][inyectado["attributes"].isna()] = inyectado["attribute"]
    inyectado["attributes"] = inyectado["attributes"].str.replace("{", "").str.replace("}", "").str.replace("'", "").str.replace(": ", ">").str.replace(":", ">").str.replace(",", "-")
    inyectado = inyectado[["business_id", "attributes"]]
    df_similares["master"] = df_similares["master"].merge(aplano_listas(yelp_set, inyectado, "attribute", "attribute", ["attributes"]), on="business_id", how='left')
    df_similares["master"]["hours"] = df_similares["master"]["hours"].str.replace("{", "").str.replace("}", "").str.replace("'", "").str.replace(": ", ">").str.replace(":", ">").str.replace(",", "-")
    df_similares["master"].drop(['postal_code_id','latitude', 'longitude', 'stars', 'review_count'], axis=1, inplace=True)
    # con ese último set filtramos los reviews y agrupamos por empresa calculando la suma y la cantidad de sentiment_score
    sentimientos = df["reviews_yelp"][df["reviews_yelp"]["business_id"].isin(yelp_set)].groupby("business_id").agg({"sentiment_score": ["sum","count"]})

    # calculamos el promedio de sentiment_score y lo guardamos en una nueva columna
    sentimientos.columns =sentimientos.columns.droplevel(0)
    sentimientos['avg_score_sentimients'] = sentimientos['sum'] / sentimientos['count']

    sentimientos.drop(["sum", "count"], axis=1, inplace=True)

    sentimientos = sentimientos.reset_index()
    df_similares["master"] = df_similares["master"].merge(sentimientos, on="business_id", how='left')
    df_similares["master"]["id"] = df_similares["master"].index + 1
    df_similares["master"]["origen"] = 'Yelp'
    df_similares["master"].reset_index(drop=True, inplace=True)
    df_similares["master"] = df_similares["master"][['business_id', 'id', 'name', 'address', 'avg_score_sentimients', 'city_id', 'hours', 'categoy',
        'attribute', 'origen']]
    
    ####### GOOGLE
    business_similar_google = df["business_google"]
    google_set = set(business_similar_google["gmap_id"])
    inyectado = df["busimess_category_google"].merge(df["category_google"], on="category_id")
    business_similar_google = business_similar_google.merge(aplano_listas_google(google_set, inyectado, "categoy", "category", ["category"]), on="gmap_id", how='left')
    inyectado = df["business_misc_google"].merge(df["misc_google"], on="misc_id")
    inyectado["attributes"] = inyectado["misc"] + ">" + inyectado["value"]
    business_similar_google = business_similar_google.merge(aplano_listas_google(google_set, inyectado, "attribute", "attribute", ["attributes"]), on="gmap_id", how='left')
    inyectado = df["business_horarios_google"]
    inyectado["hours"] = inyectado["day"].astype(str)
    inyectado["hours"] = inyectado["hours"].str.replace("1", "Sunday'").str.replace("2", "Monday").str.replace("3", "Tuesday").str.replace("4", "Wednesday").str.replace("5", "Thursday").str.replace("6", "Friday").str.replace("7", "Saturday") + ">" + inyectado["open"] + ">" + inyectado["close"]
    business_similar_google = business_similar_google.merge(aplano_listas_google(google_set, inyectado, "hours", "hours", ["hours"]), on="gmap_id", how='left')
    business_similar_google.reset_index(drop=True, inplace=True)
    business_similar_google["hours"] = business_similar_google["hours"].str.replace(",", "-")
    business_similar_google.head()
    business_similar_google.columns
    business_similar_google.drop(['description', 'postal_code_id', 'latitude', 'longitude', 'num_of_reviews', 'avg_rating'], axis=1, inplace=True)
    business_similar_google.rename(columns={'attribute_x': 'attribute'}, inplace=True)
    # con ese último set filtramos los reviews y agrupamos por empresa calculando la suma y la cantidad de sentiment_score
    sentimientos = df["reviews_google"][df["reviews_google"]["gmap_id"].isin(google_set)].groupby("gmap_id").agg({"sentiment_score": ["sum","count"]})

    # calculamos el promedio de sentiment_score y lo guardamos en una nueva columna
    sentimientos.columns =sentimientos.columns.droplevel(0)
    sentimientos['avg_score_sentimients'] = sentimientos['sum'] / sentimientos['count']

    sentimientos.drop(["sum", "count"], axis=1, inplace=True)

    sentimientos = sentimientos.reset_index()
    business_similar_google = business_similar_google.merge(sentimientos, on="gmap_id", how='left')
    business_similar_google.rename(columns={"gmap_id": "business_id"}, inplace=True)
    business_similar_google["id"] = business_similar_google.index + 1
    business_similar_google["origen"] = 'Google'
    business_similar_google.reset_index(drop=True, inplace=True)
    business_similar_google = business_similar_google[['business_id', 'id', 'name', 'address', 'avg_score_sentimients', 'city_id', 'hours', 'categoy',
        'attribute', 'origen']]

###### UNIMOS LOS DATAFRAMES DE YELP Y GOOGLE
    df_similares["master"] = pd.concat([df_similares["master"], business_similar_google]).sort_values(by='avg_score_sentimients', ascending=False).reset_index(drop=True)
    return

def entrena_business_similares(message):
    usuarios[message.chat.id]["similares"] = df_similares["master"][df_similares["master"]["city_id"] == usuarios[message.chat.id]["city_id"]]

    usuarios[message.chat.id]["similares"].reset_index(drop=True, inplace=True)
    usuarios[message.chat.id]["similares"]["id"] = usuarios[message.chat.id]["similares"].index +1

    usuarios[message.chat.id]["similares"]["id"] = usuarios[message.chat.id]["similares"]["id"].reset_index(drop=True)

    caracteristicas = ['id', 'name', 'address', 'hours', 'categoy', 'attribute', 'avg_score_sentimients']
    usuarios[message.chat.id]["filtro"] = usuarios[message.chat.id]["similares"][caracteristicas]
    usuarios[message.chat.id]["filtro"][caracteristicas] = usuarios[message.chat.id]["filtro"][caracteristicas].astype(str)
    for caracteristica in caracteristicas:
        usuarios[message.chat.id]["filtro"][caracteristica] = usuarios[message.chat.id]["filtro"][caracteristica].apply(lambda x: x.lower().replace(" ",""))
    def crear_sopa(dato):
        return dato['name'] + ' ' + dato['address'] + ' ' + dato['hours'] + ' ' + dato['categoy'] + ' ' + dato['attribute'] + ' ' + dato['avg_score_sentimients']
    usuarios[message.chat.id]["filtro"]["caracteristicas"] = usuarios[message.chat.id]["filtro"].apply(crear_sopa, axis=1)
    from sklearn.feature_extraction.text import TfidfVectorizer

    modelo = TfidfVectorizer(stop_words="english")
    matriz_del_modelo = modelo.fit_transform(usuarios[message.chat.id]["filtro"]["caracteristicas"])

    from sklearn.metrics.pairwise import cosine_similarity
    usuarios[message.chat.id]["similitud_del_coseno"] = cosine_similarity(X=matriz_del_modelo, Y=matriz_del_modelo)

    usuarios[message.chat.id]["filtro"] = usuarios[message.chat.id]["filtro"].reset_index()
    usuarios[message.chat.id]["indice"] = pd.Series(usuarios[message.chat.id]["filtro"].index, index=usuarios[message.chat.id]["filtro"]["id"])
    return


def obtener_recomendaciones_similares(message):
    id = usuarios[message.chat.id]["similares"][usuarios[message.chat.id]["similares"]["business_id"] == usuarios[message.chat.id]["similar_a"]]["id"].values[0]

    idx = usuarios[message.chat.id]["indice"][id]
    puntajes_similares = list(enumerate(usuarios[message.chat.id]["similitud_del_coseno"][idx]))
    puntajes_similares = sorted(puntajes_similares, key = lambda x: x[1], reverse=True)
    puntajes_similares = puntajes_similares[1:11]
    sitios_indices = [int(i[0]) for i in puntajes_similares]

    return usuarios[message.chat.id]["similares"].loc[sitios_indices, ['business_id', 'name', 'address', 'avg_score_sentimients', 'city_id', 'origen']]


##### Creamos un procedimiento que crea o lee el dataframe para el modelo de ML de recomendación de sitios similares
# Definimos la caducidad en horas
caducidad_horas = 8760  # Por ejemplo, 24 horas

# Obtener la fecha y hora actual
fecha_actual = datetime.now()

# Obtener la fecha y hora límite (actual - caducidad)
fecha_limite = fecha_actual - timedelta(hours=caducidad_horas)

# Obtener la lista de archivos en el directorio
archivos = ['parquet/ml_similares.parquet']  # Sustituye con tus nombres de archivo

# Filtrar archivos que cumplen con el criterio de fecha
archivos_validos = [archivo for archivo in archivos if os.path.isfile(archivo) and os.path.getmtime(archivo) > fecha_limite.timestamp()]

# Verificar fecha archivo similares
if len(archivos_validos) >= len(archivos):
    # Leer los archivos
    for archivo_valido in archivos_validos:
        df_similares["master"] = pd.read_parquet(archivo_valido)
else:
    # Ejecutar otro proceso
    print("No hay archivo de similitud válido se generará")
    genera_business_similar()
    df_similares["master"].to_parquet(r"parquet/ml_similares.parquet", engine='pyarrow', compression='snappy')

# ****************************************************************************************************************************** #

#### BOT
# iniciamos la instancia del bot
bot = telebot.TeleBot("6738398782:AAGKxjMmSKrkhNRQtoYdP8le_z3dJHQsvb4")

# Definimos las funciones para los comandos del bot start
@bot.message_handler(commands=["start"])
def cmd_bienvenida(message):
    ## Enviamos una imagen de portada
    # Envía la imagen como archivo binario
    bot.send_chat_action(message.chat.id, "upload_photo", timeout=1)
    with open('Imagenes/portada_bot.jpg', 'rb') as foto:
        bot.send_photo(message.chat.id, foto)

    # Damos la bienvenida al usuario
    markup = ReplyKeyboardRemove()
    bot.send_chat_action(message.chat.id, "typing", timeout=2)
    bot.send_message(message.chat.id, "<b>👩🏽‍🍳¡Bienvenido al Bot de Recomendaciones Gastronómicas!</b>", parse_mode="html", disable_web_page_preview=True, reply_markup=markup)

    # Guardamos el nombre del usuario para ese chat.id
    usuarios[message.chat.id] = {}
    usuarios[message.chat.id]["nombre"] = message.from_user.first_name

    #Damos la bienveninida
    bot.send_chat_action(message.chat.id, "typing", timeout=2)
    bot.send_message(message.chat.id, f"😀¡Qué bueno tenerte aquí <b>{message.from_user.first_name}</b>!", parse_mode="html", disable_web_page_preview=True)

    preguntar_modo(message)

def preguntar_modo(message):
    # Explicamos los 2 modos de recomendaciones

    bot.send_chat_action(message.chat.id, "typing", timeout=1)
    msg = bot.send_message(message.chat.id, "ℹ️<b>Tenemos dos modos en que podemos recomendarte sitios de gastronomía:</b>\n✅Sugiriéndote los mejores <u><b>sitios similares</b></u> a uno que te guste\n✅Seleccionando para tí los mejores lugares según los <u><b>tipos de sitios</b></u> de tu preferencia", parse_mode="html", disable_web_page_preview=True)

    # Definimos los dos botones para el modo de recomendación
    markup = ReplyKeyboardMarkup(
        one_time_keyboard=True,
        input_field_placeholder= "💥Presiona un Botón",
        resize_keyboard=True
    )
    markup.row("👬🏻Sitios similares") # Cargamos el texto de los botones
    markup.row("🔖Tipos de sitios") # Cargamos el texto de los botones
    markup.row("🙋🏽‍♂️Creditos") # Cargamos el texto de los botones

    #Preguntamos por el modo de recomendación
    bot.send_chat_action(message.chat.id, "typing", timeout=1)
    msg = bot.send_message(message.chat.id, "👉<b>Selecciona un botón con el Modo de Recomendación que prefieres</b>", parse_mode="html", disable_web_page_preview=True, reply_markup=markup)
    bot.register_next_step_handler(msg, direccionar_modos)

def direccionar_modos(message):
    if message.text == "🙋🏽‍♂️Creditos":
        bot.send_chat_action(message.chat.id, "typing", timeout=1)
        bot.send_message(message.chat.id, f"👋<b>¡Gracias por tu visita {message.from_user.first_name}! Esperamos que tu experiencia sea agradable y útil.</b>\n❤️No olvides recomendarnos", parse_mode="html", disable_web_page_preview=True)
        bot.send_message(message.chat.id, f"✍️<b>Bot desarrollado en el entorno educativo de 🎓Henry® por los alumnos de la carrera de Data Science:\n  ✅Adalber Conde Lucero\n  ✅Carlos Eduardo Peña Niño\n  ✅David Daniel Gonzalez Seija\n  ✅Lucas Santos Oliveira\n  ✅Edgar Barbero</b>", parse_mode="html", disable_web_page_preview=True)

        markup = ReplyKeyboardMarkup(
            one_time_keyboard=True,
            input_field_placeholder= "💥Presiona el Botón",
            resize_keyboard=True
        )
        markup.row("↩️Regresar") # Cargamos el texto de los botones
        msg = bot.send_message(message.chat.id, "👉<b>Click en el botón para regresar</b>", parse_mode="html", disable_web_page_preview=True, reply_markup=markup)
        bot.register_next_step_handler(message, preguntar_modo)
    elif message.text != "👬🏻Sitios similares" and message.text != "🔖Tipos de sitios":
        bot.send_chat_action(message.chat.id, "typing", timeout=1)
        msg = bot.send_message(message.chat.id, "✋<u>ERROR</u>: Modo de Recomendación no válido\n<b>Selecciona un botón</b>", parse_mode="html", disable_web_page_preview=True)
        bot.register_next_step_handler(message, preguntar_modo) #Volvemos a invocar a esta misma función
    else:
        # Guardamos el modo para ese chat.id
        usuarios[message.chat.id]["modo"] = message.text[1:].lower()

        bot.send_chat_action(message.chat.id, "typing", timeout=1)
        bot.send_message(message.chat.id, f"👌<b>¡Genial {usuarios[message.chat.id]['nombre']}!\n⚡Haz elegido el modo de recomendación por <u>{usuarios[message.chat.id]['modo'].lower()}</u></b>", parse_mode="html", disable_web_page_preview=True)#, reply_markup=markup)

        # Definimos los botones de los estados
        markup = ReplyKeyboardMarkup(
            one_time_keyboard=True,
            input_field_placeholder= "💥Presiona un Botón",
            resize_keyboard=True
        )
        markup.row("California") # Cargamos el texto de los botones
        markup.row("Florida") # Cargamos el texto de los botones
        markup.row("Illinois") # Cargamos el texto de los botones
        markup.row("New Jersey") # Cargamos el texto de los botones
        markup.row("Pensilvania") # Cargamos el texto de los botones

        #Preguntamos por el estado
        bot.send_chat_action(message.chat.id, "typing", timeout=1)
        msg = bot.send_message(message.chat.id, "👉<b>Selecciona un botón con el Estado</b>", parse_mode="html", disable_web_page_preview=True, reply_markup=markup)
        bot.register_next_step_handler(msg, buscar_ciudad)

def buscar_ciudad(message):
    if message.text != "California" and message.text != "Florida" and message.text != "Illinois" and message.text != "New Jersey" and message.text != "Pensilvania":
        bot.send_chat_action(message.chat.id, "typing", timeout=1)
        msg = bot.send_message(message.chat.id, "✋<u>ERROR</u>: Estado no válido\n<b>Selecciona un botón</b>", parse_mode="html", disable_web_page_preview=True)
        bot.register_next_step_handler(message, buscar_ciudad) #Volvemos a invocar a esta misma función
    else:
        # Guardamos el modo para ese chat.id
        estados = {"California": "CA", "Florida": "FL", "Illinois": "IL", "New Jersey": "NJ", "Pensilvania": "PA"}
        usuarios[message.chat.id]["estado"] = estados[message.text]
        markup = ReplyKeyboardRemove()
        msg = bot.send_message(message.chat.id, "👉<b>Ingresa el nombre de la Ciudad</b>", parse_mode="html", disable_web_page_preview=True, reply_markup=markup)
        bot.register_next_step_handler(msg, lista_ciudades) # Pasamos a la función preguntar modo el nombre

def lista_ciudades(message):
    df_ciudad = df["city"][(df["city"]["state"] == usuarios[message.chat.id]["estado"]) & (df["city"]["city"].str.contains(message.text.lower().title()))]
    if len(message.text) < 3 or len(df_ciudad) == 0:
        bot.send_chat_action(message.chat.id, "typing", timeout=1)
        msg = bot.send_message(message.chat.id, "✋<u>ERROR</u>: Ciudad no válida\n<b>Debes ingresar el Nombre de la Ciudad sobre la que deseas la recomendación</b>", parse_mode="html", disable_web_page_preview=True)
        bot.register_next_step_handler(message, lista_ciudades) #Volvemos a invocar a esta misma función
    else:
        markup = ReplyKeyboardMarkup(
        one_time_keyboard=True,
        input_field_placeholder= "💥Presiona un Botón",
        resize_keyboard=True
        )
        df_ciudad = df_ciudad.head(50)
        for index, row in df_ciudad.iterrows():
            markup.row(str(row["city_id"]) + "|" + row["city"])  # Agregamos el texto de los botones

        #Preguntamos por la confirmación de la ciudad
        bot.send_chat_action(message.chat.id, "typing", timeout=1)
        msg = bot.send_message(message.chat.id, "👉<b>Selecciona un botón confirmando la ciudad</b>", parse_mode="html", disable_web_page_preview=True, reply_markup=markup)
        bot.register_next_step_handler(msg, verify_ciudad)

def verify_ciudad(message):
    usuarios[message.chat.id]["city_id"] = message.text.split("|")[0]

    if (not usuarios[message.chat.id]["city_id"].isdigit()) or (len(df["city"][(df["city"]["city_id"] == int(usuarios[message.chat.id]["city_id"]))]) != 1):
        bot.send_chat_action(message.chat.id, "typing", timeout=1)
        msg = bot.send_message(message.chat.id, "✋<u>ERROR</u>: Ciudad no válida\n<b>Selecciona un botón</b>", parse_mode="html", disable_web_page_preview=True)
        bot.register_next_step_handler(message, verify_ciudad) #Volvemos a invocar a esta misma función
    else:
        usuarios[message.chat.id]["city_id"] = int(usuarios[message.chat.id]["city_id"])

        if usuarios[message.chat.id]["modo"] == "tipos de sitios":
            tipos_sitios(message)
        else:
            sitios_similares(message)


def tipos_sitios(message):
    # Definimos los botones de los tipos de establecimientos
    markup = ReplyKeyboardMarkup(
        one_time_keyboard=True,
        input_field_placeholder= "💥Presiona un Botón",
        resize_keyboard=True
    )

#"burgers", "cakes and desserts", "coffees and teas"
#"fast foods", "pubs/gastro-pubs", "gluten free"
#"pasta", "pizzas", "vegans/vegetarians"
#"restaurants", "others"
    markup.row("🍔Burgers", "🥮Cakes and desserts", "☕Coffees and teas") # Cargamos el texto de los botones
    markup.row("🍟Fast foods", "🍻Pubs/gastro-pubs", "✨Gluten free") # Cargamos el texto de los botones
    markup.row("🍝Pasta", "🍕Pizzas", "👩🏽‍🍳Restaurants") # Cargamos el texto de los botones
    markup.row("🥗Vegans/vegetarians", "🤔Others") # Cargamos el texto de los botones

    #Preguntamos por el tipo de establecimiento
    bot.send_chat_action(message.chat.id, "typing", timeout=1)
    msg = bot.send_message(message.chat.id, "👉<b>Selecciona un botón con el tipo de establecimiento</b>", parse_mode="html", disable_web_page_preview=True, reply_markup=markup)
    bot.register_next_step_handler(msg, filtro_tipo)

def filtro_tipo(message):
    tipos = ["🍔Burgers", "🥮Cakes and desserts", "☕Coffees and teas", "🍟Fast foods", "🍻Pubs/gastro-pubs", "✨Gluten free", "🍝Pasta", "🍕Pizzas", "👩🏽‍🍳Restaurants", "🥗Vegans/vegetarians", "🤔Others"]

    # Creamos los datadrames para los resultados
    usuarios[message.chat.id]["resultado"] = pd.DataFrame()
    usuarios[message.chat.id]["resultado_g"] = pd.DataFrame()

    if (message.text not in tipos):
        bot.send_chat_action(message.chat.id, "typing", timeout=1)
        msg = bot.send_message(message.chat.id, "✋<u>ERROR</u>: Tipo de establecimiento no válido\n<b>Selecciona un botón</b>", parse_mode="html", disable_web_page_preview=True)
        bot.register_next_step_handler(message, filtro_tipo) #Volvemos a invocar a esta misma función
    else:
        main_category_dic = {"🍔Burgers": 'burgers', "🥮Cakes and desserts":'cakes and desserts', "☕Coffees and teas": 'coffees and teas', "🍟Fast foods": 'fast foods', "🍻Pubs/gastro-pubs": 'pubs/gastro-pubs', "✨Gluten free": 'gluten free', "🍝Pasta": 'pasta', "🍕Pizzas": 'pizzas', "🥗Vegans/vegetarians": 'vegans/vegetarians', "👩🏽‍🍳Restaurants":'restaurants', "🤔Others": 'others'}
        usuarios[message.chat.id]["main_category"] = main_category_dic[message.text]

        #usuarios[message.chat.id] = {'nombre': 'Edgar', 'modo': 'tipos de sitios', 'estado': 'PA', 'city_id': 594, 'main_category': 'cakes and desserts'}

        ##### Leemos y filtramos los datos de Yelp #####
        # Creación de los sets de datos filtrados

        # Filtramos las empresas por el city_id seleccionado
        business_yelp_set = set(df["business_yelp"][df["business_yelp"]["city_id"] == usuarios[message.chat.id]["city_id"]]["business_id"])
        # Filtramos las categorías de negocios por el tipo de establecimiento elegido
        category_yelp_set = set(df["category_yelp"][df["category_yelp"]["main_category"] == usuarios[message.chat.id]["main_category"]]["category_id"])

        # Verificamos que ambos sets no estén vacíos
        if len(business_yelp_set) > 0 and len(category_yelp_set) > 0:
            # Creamos un nuevo set con las empresas que cumplen con los criterios de ambos sets anteriores
            business_yelp_set = set(df["business_category_yelp"][(df["business_category_yelp"]["business_id"].isin(business_yelp_set)) & (df["business_category_yelp"]["category_id"].isin(category_yelp_set))]["business_id"])

            # con ese último set filtramos los reviews y agrupamos por empresa calculando la suma y la cantidad de sentiment_score
            usuarios[message.chat.id]["resultado"] = df["reviews_yelp"][df["reviews_yelp"]["business_id"].isin(business_yelp_set)].groupby("business_id").agg({"sentiment_score": ["sum","count"]})

            # calculamos el promedio de sentiment_score y lo guardamos en una nueva columna
            usuarios[message.chat.id]["resultado"].columns = usuarios[message.chat.id]["resultado"].columns.droplevel(0)
            usuarios[message.chat.id]["resultado"]['avg_score_sentimients'] = usuarios[message.chat.id]["resultado"]['sum'] / usuarios[message.chat.id]["resultado"]['count']

            # ordenamos y tomamos los 5 negocios que tienen mayor promedio de sentiment_score
            usuarios[message.chat.id]["resultado"] = usuarios[message.chat.id]["resultado"].sort_values(by='avg_score_sentimients', ascending=False).head(5).drop(["sum", "count",], axis=1)
            # reseteamos el índice
            usuarios[message.chat.id]["resultado"] = usuarios[message.chat.id]["resultado"].reset_index()

            # traemos datos de la empresa mediante un merge
            usuarios[message.chat.id]["resultado"] = usuarios[message.chat.id]["resultado"].merge(df["business_yelp"][["business_id", "name", "address"]], on="business_id")

            # agredamos la columna que estabrece la fuente de recomendación
            usuarios[message.chat.id]["resultado"] ["recommended_by"] = "Yelp"
            # renombramos el business_id como id para unificar nombres con el dataframe que se generará con los negocios de google
            usuarios[message.chat.id]["resultado"].rename(columns={"business_id": "id"}, inplace=True)


        ##### Leemos y filtramos los datos de Google #####
        # Creación de los sets de datos filtrados

        # Creamos el set de datos de las empresas filtradas por City_id
        business_google_set = set(df["business_google"][df["business_google"]["city_id"] == usuarios[message.chat.id]["city_id"]]["gmap_id"])

        # Creamos el set de datos de las categorías filtradas por el tipo de establecimiento
        category_google_set = set(df["category_google"][df["category_google"]["main_category"] == usuarios[message.chat.id]["main_category"]]["category_id"])

        # verificamos que ambos sets no estén vacíos
        if len(business_google_set) > 0 and len(category_google_set) > 0:
            # Creamos un nuevo set con las empresas que cumplen con los criterios de ambos sets anteriores
            business_google_set = set(df["busimess_category_google"][(df["busimess_category_google"]["gmap_id"].isin(business_google_set)) & (df["busimess_category_google"]["category_id"].isin(category_google_set))]["gmap_id"])

            # con ese último set filtramos los reviews y agrupamos por empresa calculando la suma y la cantidad de sentiment_score
            usuarios[message.chat.id]["resultado_g"] = df["reviews_google"][df["reviews_google"]["gmap_id"].isin(business_google_set)].groupby("gmap_id").agg({"sentiment_score": ["sum","count"]})

            # calculamos el promedio de sentiment_score y lo guardamos en una nueva columna
            usuarios[message.chat.id]["resultado_g"].columns = usuarios[message.chat.id]["resultado_g"].columns.droplevel(0)
            usuarios[message.chat.id]["resultado_g"]['avg_score_sentimients'] = usuarios[message.chat.id]["resultado_g"]['sum'] / usuarios[message.chat.id]["resultado_g"]['count']

            # ordenamos y tomamos los 5 negocios que tienen mayor promedio de sentiment_score
            usuarios[message.chat.id]["resultado_g"] = usuarios[message.chat.id]["resultado_g"].sort_values(by='avg_score_sentimients', ascending=False).head(5).drop(["sum", "count",], axis=1)
            # reseteamos el índice
            usuarios[message.chat.id]["resultado_g"] = usuarios[message.chat.id]["resultado_g"].reset_index()

            # traemos datos de la empresa mediante un merge
            usuarios[message.chat.id]["resultado_g"] = usuarios[message.chat.id]["resultado_g"].merge(df["business_google"][["gmap_id", "name", "address"]], on="gmap_id")

            # agredamos la columna que estabrece la fuente de recomendación
            usuarios[message.chat.id]["resultado_g"] ["recommended_by"] = "Google"

            # renombramos el business_id como id para unificar nombres con el dataframe que se generó con los negocios de Yelp
            usuarios[message.chat.id]["resultado_g"].rename(columns={"gmap_id": "id"}, inplace=True)

        # Se concatenan los dos dataframes y se reordenan las empresas por promedio de score_sentimients de modo descendente
        usuarios[message.chat.id]["resultado"] = pd.concat([usuarios[message.chat.id]["resultado"], usuarios[message.chat.id]["resultado_g"]]).sort_values(by='avg_score_sentimients', ascending=False).reset_index(drop=True)

        # Verificación de que existan resultados
        if len(usuarios[message.chat.id]["resultado"]) < 1:
            bot.send_chat_action(message.chat.id, "typing", timeout=1)
            msg = bot.send_message(message.chat.id, "✋<u>ERROR</u>: Tu consulta no arrojó resultados", parse_mode="html", disable_web_page_preview=True)

            bot.send_chat_action(message.chat.id, "typing", timeout=1)
            markup = ReplyKeyboardMarkup(
            one_time_keyboard=True,
            input_field_placeholder= "💥Presiona el Botón",
            resize_keyboard=True
            )
        

            markup.row("😉Realizar una nueva consulta")  # Agregamos el texto de los botones

            #Preguntamos por la confirmación de la ciudad
            bot.send_chat_action(message.chat.id, "typing", timeout=1)
            msg = bot.send_message(message.chat.id, f"👉<b>{usuarios[message.chat.id]['nombre']}, haz click en el botón para hacer otra consulta</b>", parse_mode="html", disable_web_page_preview=True, reply_markup=markup)
            bot.register_next_step_handler(msg, preguntar_modo)

        else:
            markup = ReplyKeyboardMarkup(
            one_time_keyboard=True,
            input_field_placeholder= "💥Presiona un Botón",
            resize_keyboard=True
            )

            # Creamos los botones para cada una de las empresas
            # columnas del dataframe 'id', 'avg_score_sentimients', 'name', 'address', 'recommended_by'
            usuarios[message.chat.id]["resultado"] = usuarios[message.chat.id]["resultado"].head(50)
            usuarios[message.chat.id]["resultado"].set_index('id', inplace=True)

            iconos = {'burgers': '🍔', 'cakes and desserts': '🥮', 'coffees and teas': '☕', 'fast foods': '🍟', 'pubs/gastro-pubs': '🍻', 'gluten free': '✨', 'pasta': '🍝', 'pizzas': '🍕', 'restaurants': '👩🏽‍🍳', 'vegans/vegetarians': '🥗', 'others': '🤔'}
            for index, row in usuarios[message.chat.id]["resultado"].iterrows():
                #markup.row(iconos[usuarios[message.chat.id]["main_category"]] + row["name"] + " | " + row["address"] + " | " + str(round((5*(1 + row["avg_score_sentimients"])) / 2, 2)).replace(".", ",") + "⭐"  + " | " + row["recommended_by"])  # Agregamos el texto de los botones
                markup.row(f"{iconos[usuarios[message.chat.id]['main_category']]} {row['name']} | {row['address']} | {str(round((5*(1 + row['avg_score_sentimients'])) / 2, 2)).replace('.', ',')}⭐ | {row['recommended_by']}")  # Agregamos el texto de los botones

            #Preguntamos por la confirmación de la ciudad
            bot.send_chat_action(message.chat.id, "typing", timeout=1)
            msg = bot.send_message(message.chat.id, "👉<b>Selecciona un botón para ver más detalles de un negocio</b>", parse_mode="html", disable_web_page_preview=True, reply_markup=markup)
            bot.register_next_step_handler(msg, mostrar_negocio)

def mostrar_negocio(message):

    iconos = {'burgers': '🍔', 'cakes and desserts': '🥮', 'coffees and teas': '☕', 'fast foods': '🍟', 'pubs/gastro-pubs': '🍻', 'gluten free': '✨', 'pasta': '🍝', 'pizzas': '🍕', 'restaurants': '👩🏽‍🍳', 'vegans/vegetarians': '🥗', 'others': '🤔'}
    nombre = message.text.split(" | ")[0].replace(iconos[usuarios[message.chat.id]['main_category']], "").strip()

    id = message.chat.id, usuarios[message.chat.id]["resultado"][usuarios[message.chat.id]["resultado"]["name"] == nombre].index[0]

    usuarios[message.chat.id]["resultado"] = usuarios[message.chat.id]["resultado"][usuarios[message.chat.id]["resultado"]["name"] == nombre]

    nombre = usuarios[message.chat.id]["resultado"]["name"]
    direccion = usuarios[message.chat.id]["resultado"]["address"] + " - "+ devolver_ciudad(usuarios[message.chat.id]["city_id"])
    sentimiento = str(round((5 * (1 + usuarios[message.chat.id]["resultado"]["avg_score_sentimients"].values[0])) / 2, 2)).replace(".", ",")

    texto = '<u>Nombre</u>: <b>' + nombre + '</b>\n<u>Dirección</u>: <b>' + direccion + '</b>'
    if pd.notna(usuarios[message.chat.id]["resultado"]["avg_score_sentimients"].values[0]):
        texto += '\n<u>Puntuación</u>: ' + sentimiento + '⭐'

    if usuarios[message.chat.id]["resultado"]["recommended_by"].values[0] == "Google":

        usuarios[message.chat.id]["horarios"] = df["business_horarios_google"][df["business_horarios_google"]["gmap_id"] == id]
        if len(usuarios[message.chat.id]["horarios"]) > 0:
            texto = texto + "\n\n<u>Horarios:</u>"
            dias = {1: "Domingo", 2: "Lunes", 3: "Martes", 4: "Miércoles", 5: "Jueves", 6: "Viernes", 7: "Sábado"}

            usuarios[message.chat.id]["horarios"].sort_values(by='day', ascending=False)
            for index, row in usuarios[message.chat.id]["horarios"].iterrows():
                texto += "\n<b>" + dias[row["day"]] + ":</b> de " + row["open"] + " a " + row["close"]

        usuarios[message.chat.id]["misc"] = df["business_misc_google"][df["business_misc_google"]["gmap_id"] == id]
        if len(usuarios[message.chat.id]["misc"]) > 0:
            usuarios[message.chat.id]["misc"] = usuarios[message.chat.id]["misc"].merge(df["misc_google"], on="misc_id")
            texto = texto + "\n\n<u>Otras Características:</u>"

            for index, row in usuarios[message.chat.id]["misc"].iterrows():
                if index == 0:
                    texto += "<b>\n" + row["misc"] + ":</b> " + row["value"]
                texto += ", <b>" + row["misc"] + ":</b> " + row["value"]

    else:

        usuarios[message.chat.id]["horarios"] = df["business_yelp"][df["business_yelp"]["business_id"] == id]["hours"]
        if len(usuarios[message.chat.id]["horarios"]) > 0:
            lista_horario = []

            usuarios[message.chat.id]["horarios"] = re.sub(r"'(.*?)': '([^']+)'", r'"\1": "\2",', usuarios[message.chat.id]["horarios"])
            usuarios[message.chat.id]["horarios"] = usuarios[message.chat.id]["horarios"].rstrip(',')

            try:
                usuarios[message.chat.id]["horarios"] = json.loads(usuarios[message.chat.id]["horarios"])
                texto = texto + "\n\n<u>Horarios:</u>"
                dias = {'Sunday': 'Domingo', 'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles', 'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado'}
                for day,hours in usuarios[message.chat.id]["horarios"].items():
                    texto += "\n<b>" + dias[day] + ":</b> " + hours
            except json.JSONDecodeError as e:
                print(f"Error de JSON: {e}")

        usuarios[message.chat.id]["attribute"] = df["attribute_yelp"][df["business_attribute_yelp"]["business_id"] == id]
        if len(usuarios[message.chat.id]["attribute"]) > 0:
            usuarios[message.chat.id]["attribute"] = usuarios[message.chat.id]["attribute"].merge(df["attribute_yelp"], on="attribute_id")
            texto = texto + "\n\n<u>Otras Características:</u>"

            for index, row in usuarios[message.chat.id]["attribute"].iterrows():
                if index == 0:
                    texto += "<b>\n" + row[""] + "</b>"
                texto += ", <b>" + row["attribute"] + "</b> "

    bot.send_chat_action(message.chat.id, "typing", timeout=1)
    bot.send_message(message.chat.id, texto, parse_mode="html")

    bot.send_chat_action(message.chat.id, "typing", timeout=1)
    markup = ReplyKeyboardMarkup(
    one_time_keyboard=True,
    input_field_placeholder= "💥Presiona el Botón",
    resize_keyboard=True
    )

    markup.row("😉Realizar una nueva consulta")  # Agregamos el texto de los botones

    #Preguntamos por la confirmación de la ciudad
    bot.send_chat_action(message.chat.id, "typing", timeout=1)
    msg = bot.send_message(message.chat.id, f"👉<b>{usuarios[message.chat.id]['nombre']}, haz click en el botón para hacer otra consulta</b>", parse_mode="html", disable_web_page_preview=True, reply_markup=markup)
    bot.register_next_step_handler(msg, preguntar_modo)

def devolver_ciudad(ciudad_id):
    estados = {"CA": "California", "FL": "Florida", "IL": "Illinois", "PA": "Pensilvania", "NJ": "New Jersey"}
    ciudad_filtrada = df["city"][df["city"]["city_id"] == ciudad_id]

    ciudad = ciudad_filtrada.iloc[0]["city"]
    estado = estados.get(ciudad_filtrada.iloc[0]["state"], "Desconocido")
    country = ciudad_filtrada.iloc[0]["country"]
    retorno = f"{ciudad}, {estado} ({country})"

    return retorno


def sitios_similares(message):
    markup = ReplyKeyboardRemove()
    msg = bot.send_message(message.chat.id, "👉<b>Ingresa el nombre de tu sitio de referencia</b>", parse_mode="html", disable_web_page_preview=True, reply_markup=markup)
    bot.register_next_step_handler(msg, lista_sitios_similares) # Pasamos a la función preguntar modo el nombre


def lista_sitios_similares(message):
    usuarios[message.chat.id]["resultado_base"] = df_similares["master"][(df_similares["master"]["city_id"] == usuarios[message.chat.id]["city_id"]) & (df_similares["master"]["name"].str.lower().str.contains(message.text.lower()))]

    if (len(message.text) < 3) or (len(usuarios[message.chat.id]["resultado_base"]) == 0):
        bot.send_chat_action(message.chat.id, "typing", timeout=1)
        msg = bot.send_message(message.chat.id, "✋<u>ERROR</u>: sitio no válido\n<b>Debes ingresar el nombre del sitio sobre la que deseas la recomendación</b>", parse_mode="html", disable_web_page_preview=True)
        bot.register_next_step_handler(message, lista_sitios_similares) #Volvemos a invocar a esta misma función
    else:
        markup = ReplyKeyboardMarkup(
        one_time_keyboard=True,
        input_field_placeholder= "💥Presiona un Botón",
        resize_keyboard=True
        )

        # Creamos los botones para cada una de las empresas
        # columnas del dataframe 'id', 'avg_score_sentimients', 'name', 'address', 'recommended_by'
        usuarios[message.chat.id]["resultado_base"] = usuarios[message.chat.id]["resultado_base"].head(50)
        for index, row in usuarios[message.chat.id]["resultado_base"].iterrows():
            markup.row(f"👩🏽‍🍳{row['name']} | {row['address']}")  # Agregamos el texto de los botones

        #Preguntamos por la confirmación del sitio
        bot.send_chat_action(message.chat.id, "typing", timeout=1)
        msg = bot.send_message(message.chat.id, "👉<b>Selecciona un botón confirmando el sitio</b>", parse_mode="html", disable_web_page_preview=True, reply_markup=markup)
        bot.register_next_step_handler(msg, verify_sitio)

def verify_sitio(message):
    nombre = message.text.split(" | ")[0].replace("👩🏽‍🍳", "").strip()

    usuarios[message.chat.id]["similar_a"] = usuarios[message.chat.id]["resultado_base"][usuarios[message.chat.id]["resultado_base"]["name"] == nombre]["business_id"].values[0]

    if len(usuarios[message.chat.id]["resultado_base"][(usuarios[message.chat.id]["resultado_base"]["business_id"] == usuarios[message.chat.id]["similar_a"])]) != 1:
        bot.send_chat_action(message.chat.id, "typing", timeout=1)
        msg = bot.send_message(message.chat.id, "✋<u>ERROR</u>: Sitio no válido\n<b>Selecciona un botón</b>", parse_mode="html", disable_web_page_preview=True)
        bot.register_next_step_handler(message, verify_sitio) #Volvemos a invocar a esta misma función
    else:
        # Ejecutanos la función de entrenamiento del modelo de recomendación
        bot.send_message(message.chat.id, "🤯<u>ESPERA UN MOMENTO...</u> nuestra Inteligencia Artificial está trabajando en tu recomendación...", parse_mode="html", disable_web_page_preview=True)
        entrena_business_similares(message)
        
        # Ejecutamos ha función de recomendación
        usuarios[message.chat.id]["resultado"] = obtener_recomendaciones_similares(message)

        if len(usuarios[message.chat.id]["resultado"]) < 1:
            bot.send_chat_action(message.chat.id, "typing", timeout=1)
            msg = bot.send_message(message.chat.id, "✋<u>ERROR</u>: Tu consulta no arrojó resultados", parse_mode="html", disable_web_page_preview=True)

            bot.send_chat_action(message.chat.id, "typing", timeout=1)
            markup = ReplyKeyboardMarkup(
            one_time_keyboard=True,
            input_field_placeholder= "💥Presiona el Botón",
            resize_keyboard=True
            )

            markup.row("😉Realizar una nueva consulta")  # Agregamos el texto de los botones

            #Preguntamos por la confirmación de la ciudad
            bot.send_chat_action(message.chat.id, "typing", timeout=1)
            msg = bot.send_message(message.chat.id, f"👉<b>{usuarios[message.chat.id]['nombre']}, haz click en el botón para hacer otra consulta</b>", parse_mode="html", disable_web_page_preview=True, reply_markup=markup)
            bot.register_next_step_handler(msg, preguntar_modo)
        else:
            markup = ReplyKeyboardMarkup(
            one_time_keyboard=True,
            input_field_placeholder= "💥Presiona un Botón",
            resize_keyboard=True
            )            
            # Creamos los botones para cada una de las empresas
            usuarios[message.chat.id]['main_category'] = 'restaurants'

            usuarios[message.chat.id]["resultado"].rename(columns= {"origen": "recommended_by"}, inplace=True)
            usuarios[message.chat.id]["resultado"].set_index('business_id', inplace=True)

            for index, row in usuarios[message.chat.id]["resultado"].iterrows():
                texto = f"👩🏽‍🍳 {row['name']} | {row['address']}"
                if pd.notna(row["avg_score_sentimients"]):
                    texto += f" | {str(round((5*(1 + row['avg_score_sentimients'])) / 2, 2)).replace('.', ',')}⭐"
                texto += f" | {row['recommended_by']}"
                markup.row(texto)  # Agregamos el texto de los botones

            #Preguntamos por la confirmación de la la recomendación
            bot.send_chat_action(message.chat.id, "typing", timeout=1)
            msg = bot.send_message(message.chat.id, "👉<b>Selecciona un botón para ver más detalles de un negocio</b>", parse_mode="html", disable_web_page_preview=True, reply_markup=markup)
            bot.register_next_step_handler(msg, mostrar_negocio)

def leer_mensajes():
#    bot.set_my_commands([
#        telebot.types.BotCommand("/start", "Inicio y bienvenida al bot"),
#        ])
    bot.set_my_commands([])

    # Bucle infinito que comprueba permanentemente si hay mensajes nuevos en telegram
    bot.infinity_polling()
    return

# MAIN
if __name__ == '__main__':
    print("Iniciando el bot...")

    leer_mensajes()
