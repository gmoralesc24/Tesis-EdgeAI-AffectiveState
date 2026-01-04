import cv2
import time
import os
import argparse
from datetime import datetime

class VideoCapturePipeline:
    def __init__(self, output_dir="data/raw", subject_id="S001", fps=30, resolution=(640, 480)):
        """
        Inicializa el pipeline de captura de video.
        
        Args:
            output_dir (str): Directorio donde se guardarán los videos.
            subject_id (str): Identificador del sujeto (e.g., S001).
            fps (int): Frames por segundo deseados.
            resolution (tuple): Resolución del video (ancho, alto).
        """
        self.output_dir = output_dir
        self.subject_id = subject_id
        self.fps = fps
        self.width, self.height = resolution
        self.is_recording = False
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)

    def start_capture(self, camera_index=0, duration_sec=10):
        """
        Inicia la captura de video desde la cámara web.
        
        Args:
            camera_index (int): Índice de la cámara (0 usualmente).
            duration_sec (int): Duración del clip de video en segundos.
        """
        cap = cv2.VideoCapture(camera_index)
        
        # Configurar cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            return

        # Nombre del archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.subject_id}_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        # Definir codec y VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # O usar 'XVID'
        out = cv2.VideoWriter(filepath, fourcc, self.fps, (self.width, self.height))
        
        print(f"Iniciando grabación: {filepath} ({duration_sec}s)")
        start_time = time.time()
        frame_count = 0
        
        while int(time.time() - start_time) < duration_sec:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se puede recibir frame.")
                break
            
            # Escribir frame
            out.write(frame)
            
            # Mostrar frame (opcional, para feedback visual)
            cv2.imshow('Captura de Datos - Presiona Q para salir', frame)
            
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        real_fps = frame_count / (time.time() - start_time)
        print(f"Grabación finalizada. Frames totales: {frame_count}. FPS Real: {real_fps:.2f}")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Herramienta de Captura de Video para Dataset")
    parser.add_argument("--id", type=str, default="S001", help="ID del sujeto")
    parser.add_argument("--dir", type=str, default="data/raw_videos", help="Directorio de salida")
    parser.add_argument("--time", type=int, default=10, help="Duración en segundos")
    
    args = parser.parse_args()
    
    pipeline = VideoCapturePipeline(output_dir=args.dir, subject_id=args.id)
    pipeline.start_capture(duration_sec=args.time)
