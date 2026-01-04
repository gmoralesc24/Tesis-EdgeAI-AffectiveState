# Guía de Ejecución Híbrida: Google Colab vs Local
**Proyecto:** Medición Automática de Estados Afectivos (Edge AI)

Esta guía explica paso a paso cómo trabajar el proyecto utilizando la potencia de la nube (**Google Colab**) para las tareas pesadas (entrenamiento) y tu PC (**Local**) para la validación y el despliegue final.

---

## 1. Flujo de Trabajo Recomendado

| Fase | Tarea | Entorno Recomendado | ¿Por qué? |
| :--- | :--- | :--- | :--- |
| **OE1** | Descarga de Datos | **Colab** | Descarga GBs de datos en segundos (nube a nube). |
| **OE2** | Extracción Features | **Colab** | Procesar miles de videos requiere mucha CPU. |
| **OE3** | Entrenamiento | **Colab (GPU)** | Entrenar en GPU T4 (gratis) toma minutos; en CPU local horas. |
| **OE4** | Dashboard Demo | **Local** | Necesita acceso a tu webcam en tiempo real. |
| **OE5** | Validación | **Local** | Validar que el modelo ligero corre en hardware real. |

---

## 2. Preparación Previa (Configuración Estándar)

### Paso A: Subir Proyecto a Google Drive
1.  Entra a tu Google Drive.
2.  Crea una carpeta llamada `Tesis_EdgeAI`.
3.  Sube toda la carpeta `Proy_Repo` (o el contenido de este repositorio) dentro de ella.
    *   *Ruta final:* `/content/drive/MyDrive/Tesis_EdgeAI/Proy_Repo`

### Paso B: Credenciales Kaggle (Para datasets)
1.  Entra a [Kaggle > Account](https://www.kaggle.com/me/account).
2.  En "API", dale click a **Create New API Token**.
3.  Se descargará un archivo `kaggle.json`. **Guárdalo**, lo usarás en Colab.

---

## 3. Ejecución Paso a Paso en Google Colab

### Paso 1: Abrir Notebooks
1.  Ve a [Google Colab](https://colab.research.google.com/).
2.  File > Open Notebook > Google Drive.
3.  Busca `Tesis_EdgeAI/Proy_Repo/notebooks/01_Data_Acquisition.ipynb`.

### Paso 2: Configuración Inicial (En cada Notebook)
Ejecuta la celda de montaje de Drive al inicio de cada notebook:

```python
from google.colab import drive
import os
drive.mount('/content/drive')

# Cambiar al directorio del proyecto en Drive
# Ajusta la ruta si tu carpeta se llama diferente
base_path = '/content/drive/MyDrive/Tesis_EdgeAI/Proy_Repo'
os.chdir(base_path)
print("Directorio actual:", os.getcwd())
```

### Paso 3: Descarga de Datos (Notebook 01)
1.  En la sección de descarga, Colab te pedirá subir el `kaggle.json`.
2.  Ejecuta el script. Los datos se guardarán directamente en tu Drive (`data/raw`).
    *   *Nota:* Esto consume espacio de tu Google Drive (15GB gratis). Si te quedas sin espacio, usa el almacenamiento temporal de Colab (`/content/`), pero los datos se borran al cerrar la sesión.

### Paso 4: Entrenamiento y Exportación (Notebook 03)
1.  Asegúrate de cambiar el entorno a GPU: `Entorno de ejecución > Cambiar tipo > T4 GPU`.
2.  Ejecuta el entrenamiento.
3.  Al finalizar, el notebook guardará automáticamente el modelo en `models/optimized/`.
    *   Archivo clave: `mini_xception_int8.tflite`
4.  **Verificación:** Ve a tu Google Drive y verifica que el archivo `.tflite` existe en esa carpeta.

---

## 4. Transición a Local (Para Inferencia)

Una vez tengas el modelo `.tflite` generado en Colab:

### Paso 1: Sincronización
1.  Si usaste Google Drive desktop, el archivo ya estará en tu PC.
2.  Si no, entra a Drive web y descarga solo el archivo `models/optimized/mini_xception_int8.tflite`.
3.  Colócalo en la misma ruta en tu proyecto local: `Proy_Repo/models/optimized/`.

### Paso 2: Ejecución del Dashboard
1.  Abre tu terminal (Windows) en la carpeta del proyecto.
2.  Activa el entorno: `setup_env.bat` (o `.venv\Scripts\activate`).
3.  Ejecuta el demo:
    ```bash
    streamlit run src/app/dashboard.py
    ```
4.  Selecciona el modelo "Mini-Xception (Int8)" en el menú lateral. ¡Debería funcionar usando tu webcam y el modelo entrenado en la nube!

---

## 5. Estimación de Tiempos y Recursos

| Etapa | Actividad | Tiempo Estimado (Colab GPU) | Tiempo Estimado (Local CPU) | Notas |
| :--- | :--- | :--- | :--- | :--- |
| **Setup** | Configurar entorno | 5 min | 15-20 min | Local requiere instalar librerías. |
| **OE1** | Descarga Datasets | 10 min (Cloud speed) | Varía (tu internet) | FER2013 son ~100MB. DAiSEE son GBs. |
| **OE2** | Preprocesamiento | 20-40 min | 2-4 horas | Procesar video es lento sin paralelización. |
| **OE3** | Entrenamiento (50 epochs) | **15-30 min** | **+12 horas** | **CRÍTICO: Usar Colab aquí.** |
| **OE3** | Cuantización | 2 min | 5 min | Rápido en ambos. |
| **OE4** | Inferencia/Demo | N/A (No webcam directa) | Tiempo Real (30 FPS) | Colab no accede fácil a webcam local. |

---

## 6. Comandos Clave (Resumen)

**En Colab:**
```python
!pip install -r requirements.txt
!python src/data/download_datasets.py
```

**En Local:**
```bash
setup_env.bat
streamlit run src/app/dashboard.py
```
