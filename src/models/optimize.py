import tensorflow as tf
import os
import argparse

class ModelOptimizer:
    def __init__(self, model_path, output_dir="models/optimized"):
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def convert_to_tflite(self, quantization='int8', dataset_gen=None):
        """
        Convierte un modelo Keras a TFLite con optimizaciones.
        
        Args:
            quantization (str): 'float16', 'int8', o None.
            dataset_gen (generator): Generador de datos representativos para int8.
        """
        try:
            model = tf.keras.models.load_model(self.model_path)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            algo_suffix = "fp32"
            
            if quantization == 'float16':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                algo_suffix = "fp16"
                
            elif quantization == 'int8':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                if dataset_gen:
                    print("Usando dataset representativo para cuantización int8 completa...")
                    converter.representative_dataset = dataset_gen
                    # Asegurar compatibilidad Edge TPU (opcional)
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.uint8  # o tf.int8
                    converter.inference_output_type = tf.uint8
                algo_suffix = "int8"

            tflite_model = converter.convert()
            
            # Guardar
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            output_path = os.path.join(self.output_dir, f"{model_name}_{algo_suffix}.tflite")
            
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
                
            print(f"Modelo optimizado guardado en: {output_path}")
            print(f"Tamaño original: {os.path.getsize(self.model_path) / 1024:.2f} KB")
            print(f"Tamaño TFLite: {os.path.getsize(output_path) / 1024:.2f} KB")
            
            return output_path
            
        except Exception as e:
            print(f"Error en conversión: {e}")
            return None

if __name__ == "__main__":
    # Ejemplo de uso dummy
    # Crear un modelo dummy para probar
    dummy_model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,))])
    dummy_model.save("dummy_model.h5")
    
    # Optimizar
    opt = ModelOptimizer("dummy_model.h5")
    opt.convert_to_tflite(quantization='float16')
    
    # Limpiar
    if os.path.exists("dummy_model.h5"):
        os.remove("dummy_model.h5")
