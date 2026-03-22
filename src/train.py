import tensorflow as tf
from tensorflow.keras.optimizers import Adam
# Optimizador = Actualizar los pesos de la RN
# Despues del Batch, el modelo predice, calcula el error, optimización d elos pesos 
from tensorflow.keras.losses import BinaryCrossentropy
# Fx de perdida
# Menor perdida = mejor ajuste de los datos

from .config import BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_PATH
from .data import load_malaria_splits
from .model import build_lenet
# Centralizar el modelo

def train_and_save():
    train_ds, val_ds, test_ds, _ = load_malaria_splits() # Cargar el conjunto de datos

    train_ds = (
        train_ds.shuffle(8, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds.shuffle(8, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = test_ds.batch(1)

    model = build_lenet()
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=BinaryCrossentropy(),
        metrics=["accuracy"]
    )

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
    loss, acc = model.evaluate(test_ds, verbose=1)

    model.save(MODEL_PATH)
    print(f"✅ Modelo guardado en: {MODEL_PATH}")
    print(f"✅ Test accuracy: {acc:.4f} | loss: {loss:.4f}")

if __name__ == "__main__":
    train_and_save()