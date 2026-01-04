import cv2
import mediapipe as mp
import numpy as np

class FacialFeatureExtractor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Inicializa el extractor facial basado en MediaPipe Face Mesh.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # Incluye irises
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame_bgr):
        """
        Procesa un frame y extrae landmarks faciales.
        
        Args:
            frame_bgr (np.array): Imagen BGR (opencv default).
        
        Returns:
            dict: Diccionario con features calculados y raw landmarks.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        features = {}
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Convertir a lista de (x, y, z)
            coords = []
            for lm in landmarks.landmark:
                coords.append([lm.x, lm.y, lm.z])
            features['raw_landmarks'] = np.array(coords)
            
            # Calcular features derivados (ejemplo simple)
            # EAR (Eye Aspect Ratio), Angulos de cabeza (Head Pose), etc.
            features['ear_left'] = self._calculate_ear(coords, eye='left')
            features['ear_right'] = self._calculate_ear(coords, eye='right')
            features['gaze_vector'] = self._estimate_gaze(coords)
            
            return features, results.multi_face_landmarks
        
        return None, None

    def _calculate_ear(self, landmarks, eye='left'):
        # Indices de landmarks para ojos en MediaPipe (simplificado)
        # Left Eye: 33, 160, 158, 133, 153, 144
        # Right Eye: 362, 385, 387, 263, 373, 380
        if eye == 'left':
            indices = [33, 160, 158, 133, 153, 144]
        else:
            indices = [362, 385, 387, 263, 373, 380]
            
        p = [np.array([landmarks[i][0], landmarks[i][1]]) for i in indices]
        
        # EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        v3 = np.linalg.norm(p[0] - p[3])
        
        ear = (v1 + v2) / (2.0 * v3) if v3 > 0 else 0
        return ear

    def _estimate_gaze(self, landmarks):
        # Estimación simplificada usando iris
        # MediaPipe iris indices: Left: 468, Right: 473
        left_iris = landmarks[468]
        right_iris = landmarks[473]
        
        # Calcular centro de ojos promedio vs centro de iris
        # Esto requiere calibración, retornamos posición cruda por ahora
        return {
            'left_iris': (left_iris[0], left_iris[1]),
            'right_iris': (right_iris[0], right_iris[1])
        }

    def close(self):
        self.face_mesh.close()

if __name__ == "__main__":
    extractor = FacialFeatureExtractor()
    # Testdummy
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    feat, _ = extractor.process_frame(dummy_frame)
    print("Features extraídos:", feat)
