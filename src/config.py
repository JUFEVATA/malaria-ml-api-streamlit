IM_SIZE = 224
# Define el tamaño en el que se redimensionarán todas las imagenes
# Antes de ser utilizadas por la red neuronal
# (224=Alturax224=Anchox3)RGB= Canales 

BATCH_SIZE = 32
# Usar 32 imagenes por lote
# Predicciones, Calcular error y actualizar los pesos
# Usar menos memoria, entrenamientos mas eficientes y GPU
# Es el valor clásico

LEARNING_RATE = 0.01 # Optimizador Adam
# Nos define la tasa de apredizaje del optimizador
# Nos controlar que tan son los ajustes
# Para las predicciones, para el calculo de error y optimización de los pesos
# Valores grande= Saltar errores y nos los detecte y entrenamiento puede ser inestable
# Valores pequeños= Entrenamiento lento, demora mas en aprender, puede ser mas estable

EPOCHS = 4
# Cuantas veces el modelo recorre el dataset

THRESHOLD = 0.5  # <0.5 = Parasitized (P), >=0.5 = Uninfected (U)
# Definimos el umbral 
# Clasificación Binaria
# Equilibrio= Sensibilidad y Especificidad

MODEL_PATH = "artifacts/lenet.keras"# Resultado del entrenamiento del modelo
# Definir la ruta donde se guardara el modelo entrenado
# lenet.keras= Arquitectura de la red neuronal, pesos entrenados, configuración del modelo
