import numpy as np

class FeatureFuser:
    def __init__(self):
        """
        Módulo para fusionar vectores de características faciales y posturales.
        """
        pass

    def fuse(self, facial_features, postural_features):
        """
        Concatena y normaliza features.
        
        Args:
            facial_features (dict): Salida de FacialFeatureExtractor.
            postural_features (dict): Salida de PosturalFeatureExtractor.
            
        Returns:
            np.array: Vector de características fusionado (flattened).
        """
        vector = []
        
        # 1. Features Faciales
        if facial_features:
            vector.append(facial_features.get('ear_left', 0.0))
            vector.append(facial_features.get('ear_right', 0.0))
            # Añadir Gaze (desempaquetar)
            gaze = facial_features.get('gaze_vector', {'left_iris': (0,0), 'right_iris': (0,0)})
            vector.append(gaze['left_iris'][0])
            vector.append(gaze['left_iris'][1])
            # Podríamos añadir más raw landmarks si fuera necesario, pero mantenemos vector corto
        else:
            vector.extend([0.0]*4) # Rellenar con ceros si no hay cara

        # 2. Features Posturales
        if postural_features:
            vector.append(postural_features.get('neck_inclination', 0.0))
            vector.append(postural_features.get('pose_confidence', 0.0))
        else:
            vector.extend([0.0]*2)
            
        return np.array(vector, dtype=np.float32)

if __name__ == "__main__":
    fuser = FeatureFuser()
    v = fuser.fuse({'ear_left': 0.3}, {'neck_inclination': 0.05})
    print("Vector fusionado:", v)
