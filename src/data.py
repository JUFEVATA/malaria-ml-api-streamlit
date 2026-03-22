import tensorflow as tf
import tensorflow_datasets as tfds
# Importación de nuestro dataset 

from .config import IM_SIZE
# Importar la constante IM_SIZE

def splits(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # División del dt; 3 subconjuntos
    # Entrenamiento, validación y prueba

    dataset_size = len(dataset)
    # Número total de ejemplos en el dt

    train_ds = dataset.take(int(train_ratio * dataset_size))# el uso del 80% de los datos
    val_test = dataset.skip(int(train_ratio * dataset_size))# Skip y devuelve el resto del dt
    val_ds = val_test.take(int(val_ratio * dataset_size))# Dt validación
    test_ds = val_test.skip(int(val_ratio * dataset_size))# Dt de prueba 
    return train_ds, val_ds, test_ds# 

def resize_rescale(image, label):

    image = tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255.0 # Normalización ayuda a que la red neuronal sea mas estable
    return image, label# Nos devuelve la imagen transformada

def load_malaria_splits():
    ds, info = tfds.load(
        "malaria",
        with_info=True,# Devolución de metadatos #Numero de ejemplos,# de clases y descripción 
        as_supervised=True,# Retorne (imagen, etiqueta)
        shuffle_files=True,
        split=["train"],
    )

    train_ds, val_ds, test_ds = splits(ds[0], 0.8, 0.1, 0.1)
    # dividir el dt usando la funcion split que definimos anteriormente 
    # 80% para entrenamiento
    # 10% para validación
    # 10% para prueba/test

    train_ds = train_ds.map(resize_rescale, num_parallel_calls=tf.data.AUTOTUNE)
        # Aceleración de peprocesamiento
    val_ds   = val_ds.map(resize_rescale, num_parallel_calls=tf.data.AUTOTUNE)
        # Aceleración de peprocesamiento
    test_ds  = test_ds.map(resize_rescale, num_parallel_calls=tf.data.AUTOTUNE)
        # Aceleración de peprocesamiento

    return train_ds, val_ds, test_ds, info
    # 