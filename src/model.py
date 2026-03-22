from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dense

# InputLayer= Define la forma de entrada de los datos que recibira el modelo
# Conv2d = Patrones espaciales, bordes, texturas y estructuras
# MaxPool2d = Disminución de la imgn manteniendo las caracteristicas mas importantes
# Flatten = Convierte las matrices en un vector 1D
# BatchNormalization = Normalizar la salida de una capa durante el entrenamiento / acelerar el aprendizaje
# Dense = Capa complementaria conectada al final de la red y es para la clasificación

from tensorflow.keras.models import Sequential
# Apilación de las capas una despues de otra

from .config import IM_SIZE

def build_lenet():
    model = Sequential([
        InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
    # 224,224,3
        Conv2D(6, 3, strides=1, padding="valid", activation="relu"),
        # 6= Número de filtros o detectores de caracteristicas 
        # strides= el filtro se mueve un pixel a la vez
        # padding = No se usa el padding a la imagen
        # padding = (Tecnica donde agregamos valores(generalmente son ceros))
        # controlar como cambia el tamaño de las imagenes, despues de la cnn
        BatchNormalization(),
        # Normalizar las activaciones de la capa anterior
        # Estabilizar el entrenamiento
        MaxPool2D(pool_size=2, strides=2),


        Conv2D(16, 3, strides=1, padding="valid", activation="relu"),
        # Segunda capa cnn
        # 16 filtros, bordes y lineas
        BatchNormalization(),
        MaxPool2D(pool_size=2, strides=2),

        Flatten(),# Nos permite hacer la conexión entre las neuronas
        Dense(100, activation="relu"),
        BatchNormalization(),
        Dense(10, activation="relu"),
        BatchNormalization(),
        Dense(1, activation="sigmoid"),  # binario
    ])
    return model