import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
import argparse
import os
from src.models.architectures import build_mini_xception, build_mobilenet_v3_small
# asumimos que existe un data loader o usamos tf.keras.utils.image_dataset_from_directory

def train_model(data_dir, model_type="mini_xception", epochs=20, batch_size=32):
    """
    Entrena un modelo de clasificación de estados afectivos.
    """
    # 1. Cargar Datos
    print(f"Cargando datos desde {data_dir}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        image_size=(48, 48) if model_type=="mini_xception" else (224, 224),
        batch_size=batch_size,
        color_mode='grayscale' if model_type=="mini_xception" else 'rgb'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'val'),
        image_size=(48, 48) if model_type=="mini_xception" else (224, 224),
        batch_size=batch_size,
        color_mode='grayscale' if model_type=="mini_xception" else 'rgb'
    )
    
    # 2. Construir Modelo
    if model_type == "mini_xception":
        model = build_mini_xception(num_classes=4)
        input_shape = (48, 48, 1)
    else:
        model = build_mobilenet_v3_small(num_classes=4)
        input_shape = (224, 224, 3)
        
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 3. Callbacks
    checkpoint_path = f"models/checkpoints/{model_type}_best.keras"
    os.makedirs("models/checkpoints", exist_ok=True)
    
    cbs = [
        callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy'),
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    # Data Augmentation (Solo activa durante entrenamiento)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # 4. Entrenar (Fase 1: Backbone Congelado)
    print(f"Fase 1: Entrenamiento inicial (Backbone congelado) por {epochs} épocas...")
    print(f"Dispositivo: GPU -> {tf.config.list_physical_devices('GPU')}")

    # Aplicar augmentation en el map si es posible, o usar layers dentro del modelo. 
    # Para simplicidad y performance en tf.data, lo haremos parte del modelo en architectures o aqui.
    # Dado que architectures.py retorna un modelo funcional, mejor inyectamos augmentation antes si es posible,
    # o simplemente lo dejamos si el dataset es pequeño.
    # MEJOR OPCIÓN: Fine-Tuning explícito.

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cbs
    )
    
    # 5. Fine-Tuning (Fase 2: Descongelar últimas capas)
    # Solo para MobileNet, no tiene sentido para Mini-Xception (ya es pequeño y trainable)
    if model_type == "mobilenet":
        print("\n--- Iniciando Fase 2: Fine-Tuning ---")
        base_model = model.layers[2] # Indice asumiendo: Input -> Rescaling -> MobileNet -> ...
        if isinstance(base_model, tf.keras.Model):
            base_model.trainable = True
            
            # Congelar las primeras 100 capas (MobileNetV3 tiene muchas) para no romper features básicos
            for layer in base_model.layers[:-30]: 
                layer.trainable = False
                
            model.compile(
                optimizer=optimizers.Adam(1e-5), # Learning Rate MUY bajo para afinamiento
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            total_epochs = epochs + 10 # 10 épocas extra
            history_fine = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=total_epochs,
                initial_epoch=history.epoch[-1],
                callbacks=cbs
            )
            # Combinar historias para retorno (opcional, aqui retornamos la ultima)
            history = history_fine

    # Guardar modelo final explícitamente
    model.save(f"models/checkpoints/{model_type}_final.keras")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/images")
    parser.add_argument("--model", type=str, choices=["mini_xception", "mobilenet"], default="mini_xception")
    args = parser.parse_args()
    
    # Solo ejecutar si existe el directorio, sino solo imprimimos mensaje
    if os.path.exists(args.data):
        train_model(args.data, args.model)
    else:
        print(f"Directorio de datos {args.data} no encontrado. Ejecuta preprocess.py primero.")
