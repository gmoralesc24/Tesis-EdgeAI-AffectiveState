import csv
import json
import os
from enum import Enum

class AffectiveState(Enum):
    ATENCION = 0
    DISTRACCION = 1
    FATIGA = 2
    NEUTRAL = 3

class LabelManager:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.labels_file = os.path.join(dataset_dir, "labels.csv")
        self._ensure_labels_file()

    def _ensure_labels_file(self):
        if not os.path.exists(self.labels_file):
            with open(self.labels_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["video_id", "timestamp_start", "timestamp_end", "frame_start", "frame_end", "label_id", "label_name"])

    def add_label(self, video_id, frame_start, frame_end, state: AffectiveState):
        """Añade una etiqueta para un segmento de video."""
        with open(self.labels_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                video_id, 
                "", "", # Timestamps opcionales si usamos frames
                frame_start, 
                frame_end, 
                state.value, 
                state.name
            ])
        print(f"Etiqueta guardada para {video_id}: {state.name} ({frame_start}-{frame_end})")

    def load_labels(self):
        """Carga todas las etiquetas como una lista de diccionarios."""
        labels = []
        if os.path.exists(self.labels_file):
            with open(self.labels_file, mode='r') as file:
                reader = csv.DictReader(file)
                labels = list(reader)
        return labels

if __name__ == "__main__":
    # Ejemplo de uso
    manager = LabelManager("data/processed")
    manager.add_label("S001_video1", 0, 150, AffectiveState.ATENCION)
