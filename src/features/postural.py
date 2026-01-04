import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

class PosturalFeatureExtractor:
    def __init__(self, model_path="https://tfhub.dev/google/movenet/singlepose/lightning/4"):
        """
        Inicializa el extractor postural usando MoveNet.
        
        Args:
            model_path (str): URL de TF Hub o path local al modelo TFLite.
        """
        self.model_path = model_path
        self.input_size = 192 # Lightning usa 192x192
        
        # Cargar modelo desde Hub (o local)
        print(f"Cargando modelo MoveNet desde {model_path}...")
        try:
            self.model = hub.load(model_path)
            self.movenet = self.model.signatures['serving_default']
            print("Modelo cargado exitosamente.")
        except Exception as e:
            print(f"Error cargando MoveNet: {e}")
            self.model = None

    def process_frame(self, frame_bgr):
        """
        Procesa frame y extrae keypoints corporales.
        """
        if self.model is None:
            return None
        
        # Preprocesamiento
        img = tf.image.resize_with_pad(np.expand_dims(frame_bgr, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.int32)
        
        # Inferencia
        results = self.movenet(input_image)
        keypoints = results['output_0'].numpy()[0, 0, :, :] 
        # Shape (17, 3) -> [y, x, conf]
        
        features = self._extract_derived_features(keypoints)
        return features

    def _extract_derived_features(self, keypoints):
        """
        Calcula features derivados de la postura (ángulos, inclinación).
        Keypoints map: 0:nose, 5:left_shoulder, 6:right_shoulder, ...
        """
        features = {'raw_pose': keypoints}
        
        # Ejemplo: Inclinación de la cabeza (Nariz vs Hombros)
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        
        # Calcular punto medio de hombros
        shoulder_mid_x = (left_shoulder[1] + right_shoulder[1]) / 2
        shoulder_mid_y = (left_shoulder[0] + right_shoulder[0]) / 2
        
        # Desviación horizontal de la nariz respecto al centro de hombros
        neck_inclination = nose[1] - shoulder_mid_x
        features['neck_inclination'] = neck_inclination
        
        # Confianza promedio de la detección
        features['pose_confidence'] = np.mean(keypoints[:, 2])
        
        return features

if __name__ == "__main__":
    extractor = PosturalFeatureExtractor()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    feat = extractor.process_frame(dummy_frame)
    print("Features Posturales:", feat)
