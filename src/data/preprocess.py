import cv2
import os
import glob
import argparse
import numpy as np
from src.utils.logger import logger

class DataPreprocessor:
    def __init__(self, input_dir, output_dir, target_size=(224, 224), mode="video"):
        """
        Args:
            mode (str): 'video' (extraer frames) o 'image' (copiar/redimensionar imágenes estáticas).
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.mode = mode
        
        os.makedirs(output_dir, exist_ok=True)

    def process_video(self, video_path, sample_rate=5):
        cap = cv2.VideoCapture(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        base_output_path = os.path.join(self.output_dir, video_name)
        os.makedirs(base_output_path, exist_ok=True)
        
        frame_idx = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % sample_rate == 0:
                frame_resized = cv2.resize(frame, self.target_size)
                cv2.imwrite(f"{base_output_path}/frame_{frame_idx:04d}.jpg", frame_resized)
                saved_count += 1
            frame_idx += 1
        return saved_count

    def process_image(self, image_path):
        """Procesa una imagen individual (FER2013)."""
        try:
            img = cv2.imread(image_path)
            if img is None: return False
            
            img_resized = cv2.resize(img, self.target_size)
            
            # Mantener estructura de carpetas (ej. output/train/angry/img.jpg)
            rel_path = os.path.relpath(image_path, self.input_dir)
            out_path = os.path.join(self.output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            cv2.imwrite(out_path, img_resized)
            return True
        except Exception as e:
            logger.error(f"Error procesando imagen {image_path}: {e}")
            return False

    def process_batch(self):
        if self.mode == "video":
            files = glob.glob(os.path.join(self.input_dir, "**", "*.mp4"), recursive=True)
            logger.info(f"Procesando {len(files)} videos...")
            for f in files:
                self.process_video(f)
        elif self.mode == "image":
            files = glob.glob(os.path.join(self.input_dir, "**", "*.jpg"), recursive=True)
            if not files: # Intentar png
                files = glob.glob(os.path.join(self.input_dir, "**", "*.png"), recursive=True)
            
            logger.info(f"Procesando {len(files)} imágenes...")
            for f in files:
                self.process_image(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["video", "image"], default="video")
    args = parser.parse_args()
    
    proc = DataPreprocessor(args.input, args.output, mode=args.mode)
    proc.process_batch()
