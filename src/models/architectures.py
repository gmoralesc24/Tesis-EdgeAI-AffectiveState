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
    # [FIX] Normalización: MobileNetV3 espera [-1, 1] o [0, 1]. Keras App lo suele manejar, 
    # pero para seguridad explicita añadimos Rescaling si la entrada es raw [0,255].
    # MobileNetV3 específico de Keras incluye su preprocesamiento, pero Mini-Xception NO.
    
    # Para MobileNetV3Small de Keras, la documentación dice que "inputs are expected to be float input values", 
    # pero `preprocess_input` escala a [-1, 1]. Haremos un rescale manual simple a [0, 1] que suele funcionar bien trasnfer learning.
    x = layers.Rescaling(1./255)(inputs)
    
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
