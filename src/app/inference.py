import cv2
import numpy as np
import tensorflow as tf
import time
from src.features.facial import FacialFeatureExtractor
from src.features.postural import PosturalFeatureExtractor
from src.features.fusion import FeatureFuser

class InferenceEngine:
    def __init__(self, model_path, use_tflite=False):
        """
        Motor de inferencia que coordina la extracción de features y la predicción.
        """
        self.use_tflite = use_tflite
        self.model_path = model_path
        
        # Inicializar extractores
        self.face_extractor = FacialFeatureExtractor()
        self.pose_extractor = PosturalFeatureExtractor() # Puede fallar si no hay internet/modelo local
        self.fuser = FeatureFuser()
        
        # Cargar modelo
        if use_tflite:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.model = tf.keras.models.load_model(model_path)
            
        self.labels = ["Atencion", "Distraccion", "Fatiga", "Neutral"]

    def predict(self, frame):
        """
        Realiza la inferencia sobre un frame de video.
        """
        start_time = time.time()
        
        # 1. Extraer Features
        face_feats, _ = self.face_extractor.process_frame(frame)
        pose_feats = self.pose_extractor.process_frame(frame)
        
        # 2. Fusionar
        # NOTA: Aquí simplificamos. Si el modelo es de IMAGEN (CNN), pasamos la imagen.
        # Si el modelo es de FEATURES (MLP), pasamos el vector.
        # El plan original tenía ambos enfoques. Asumiremos Arq Híbrida o CNN pura por ahora para la demo.
        # PERO, el código de entrenamiento usaba image_dataset_from_directory, implicando CNN pura.
        # Para cumplir con el requerimiento multimodal, deberíamos tener un modelo que acepte features o una CNN que haga todo.
        # Ajuste: Si el modelo espera imagen (48x48 o 224x224), preprocesamos la imagen.
        
        # Chequeo rápido de input shape
        if self.use_tflite:
            input_shape = self.input_details[0]['shape']
        else:
            input_shape = self.model.input_shape
            
        # Si input es imagen (4D tensor)
        if len(input_shape) == 4:
            target_size = (input_shape[1], input_shape[2])
            is_gray = (input_shape[3] == 1)
            
            img_processed = cv2.resize(frame, target_size)
            if is_gray:
                img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
                img_processed = np.expand_dims(img_processed, axis=-1)
            
            img_processed = np.expand_dims(img_processed, axis=0) # Batch dim
            img_processed = img_processed.astype(np.float32) / 255.0
            
            if self.use_tflite:
                self.interpreter.set_tensor(self.input_details[0]['index'], img_processed)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            else:
                prediction = self.model.predict(img_processed, verbose=0)[0]
        
        else:
            # Modelo basado en features (vector)
            feature_vector = self.fuser.fuse(face_feats, pose_feats)
            feature_vector = np.expand_dims(feature_vector, axis=0)
            
            if self.use_tflite:
                self.interpreter.set_tensor(self.input_details[0]['index'], feature_vector)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            else:
                prediction = self.model.predict(feature_vector, verbose=0)[0]

        label_idx = np.argmax(prediction)
        confidence = prediction[label_idx]
        
        latency = (time.time() - start_time) * 1000 # ms
        
        return {
            "label": self.labels[label_idx],
            "confidence": float(confidence),
            "latency_ms": latency,
            "face_data": face_feats, # Para visualizar
            "probs": prediction.tolist()
        }

if __name__ == "__main__":
    # Test dummy con modelo no existente (fallará carga, pero valida sintaxis)
    try:
        engine = InferenceEngine("model.tflite", use_tflite=True)
    except:
        print("Test: No model found.")
