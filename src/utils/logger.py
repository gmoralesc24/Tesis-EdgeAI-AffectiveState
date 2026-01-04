import logging
import os
import sys

def setup_logger(name="EdgeAI_Logger", log_file="logs/execution.log", level=logging.INFO):
    """
    Configura el logger para escribir en archivo y (opcionalmente) consola.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Asegurar que directorio existe
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Handler Archivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    
    # Handler Consola (opcional, el usuario pidió minimizar esto, 
    # así que podemos poner level más alto o quitarlo si prefiere silencio total)
    # Por seguridad dejamos errores en consola.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING) 
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Instancia global por defecto
logger = setup_logger()
