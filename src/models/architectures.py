import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_mobilenet_v3_small(input_shape=(224, 224, 3), num_classes=4):
    """
    Construye un modelo MobileNetV3 Small (pre-entrenado en ImageNet) adaptado para clasificación.
    """
    base_model = applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        minimalistic=True # Más ligero, mejor para Edge
    )
    
    # Congelar backbone inicialmente
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape)
    
    # [MEJORA] Data Augmentation integrada en el modelo (Solo activa en training=True)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    # [FIX] Normalización: MobileNetV3 espera [-1, 1] o [0, 1].
    x = layers.Rescaling(1./127.5, offset=-1)(x) # Estandar para MobileNet [-1, 1]
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="MobileNetV3_Small_Custom")
    return model

def build_mini_xception(input_shape=(48, 48, 1), num_classes=4):
    """
    Implementación de Mini-Xception.
    """
    input_layer = layers.Input(shape=input_shape)
    
    # [FIX] Normalización indispensable [0, 255] -> [0, 1]
    x = layers.Rescaling(1./255)(input_layer)
    
    x = layers.Conv2D(8, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(8, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Bloques residuales con separables convs
    previous = x
    
    # Bloque 1
    x = layers.SeparableConv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    previous = layers.Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(previous)
    x = layers.add([x, previous])
    previous = x 
    
    # Bloque 2
    x = layers.SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    previous = layers.Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(previous)
    x = layers.add([x, previous])
    
    # Salida
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(input_layer, outputs, name="Mini_Xception")
    return model

if __name__ == "__main__":
    model = build_mini_xception()
    model.summary()
