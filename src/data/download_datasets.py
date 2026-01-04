import os
import requests
import zipfile
import io
from src.utils.logger import logger

def ensure_dirs(base_dir):
    """Crea los subdirectorios para cada dataset."""
    os.makedirs(os.path.join(base_dir, "fer2013"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "daisee"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "nthu_ddd"), exist_ok=True)

def info_fer2013(output_dir):
    """Instrucciones FER2013 (Imágenes - Estado 'Otros')."""
    readme_path = os.path.join(output_dir, "fer2013", "README.txt")
    with open(readme_path, "w") as f:
        f.write("# Dataset FER2013 (Imágenes)\n")
        f.write("Uso: Estado 'Otros' (Neutral, Emociones) y pre-entrenamiento.\n")
        f.write("1. Descargar: https://www.kaggle.com/datasets/msambare/fer2013\n")
        f.write("2. Colocar 'train' y 'test' folders dentro de: data/raw/fer2013/\n")
    logger.info(f"Instrucciones FER2013 en {readme_path}")

def info_daisee(output_dir):
    """Instrucciones DAiSEE (Video - Estados 'Atención', 'Distracción')."""
    readme_path = os.path.join(output_dir, "daisee", "README.txt")
    with open(readme_path, "w") as f:
        f.write("# Dataset DAiSEE (Video)\n")
        f.write("Uso: Estado 'Atención' (High Engagement), 'Distracción' (Low Engagement).\n")
        f.write("1. Solicitar acceso: https://people.iith.ac.in/vineethnb/resources/daisee/\n")
        f.write("2. Descargar particiones (Train, Test, Validation).\n")
        f.write("3. Descomprimir en: data/raw/daisee/\n")
    logger.info(f"Instrucciones DAiSEE en {readme_path}")

def info_nthu_ddd(output_dir):
    """Instrucciones NTHU-DDD (Video - Estado 'Fatiga')."""
    readme_path = os.path.join(output_dir, "nthu_ddd", "README.txt")
    with open(readme_path, "w") as f:
        f.write("# Dataset NTHU-DDD (Driver Drowsiness Detection)\n")
        f.write("Uso: Estado 'Fatiga'. Contiene videos de somnolencia real (bostezos, cabeceos).\n")
        f.write("1. Sitio oficial: https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/\n")
        f.write("   (Si no abre, buscar 'NTHU Driver Drowsiness Detection Dataset' en paperswithcode.com)\n")
        f.write("2. Solicitar dataset a los autores.\n")
        f.write("3. Colocar videos (ej. 'yawning', 'nodding') en: data/raw/nthu_ddd/\n")
    logger.info(f"Instrucciones NTHU-DDD en {readme_path}")

def setup_datasets(base_dir="data/raw"):
    logger.info("Configurando estructura de datasets recomendados...")
    ensure_dirs(base_dir)
    info_fer2013(base_dir)
    info_daisee(base_dir)
    info_nthu_ddd(base_dir)
    logger.info("Setup de directorios e instrucciones completado.")

if __name__ == "__main__":
    setup_datasets()
