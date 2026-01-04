# Estimación de Recursos y Requerimientos Técnicos
**Proyecto:** Medición Automática de Estados Afectivos (Edge AI)

Este documento detalla los recursos de hardware y almacenamiento necesarios para desarrollar, entrenar y ejecutar el proyecto localmente.

## 1. Espacio en Disco (Almacenamiento)

La mayor carga de almacenamiento proviene de los datasets de video crudos y los frames procesados.

### Estimación por Dataset

| Dataset | Contenido | Tamaño Estimado (Raw) | Tamaño Procesado (Frames/Features) |
| :--- | :--- | :--- | :--- |
| **DAiSEE** | 9,068 videos (10s, VGA) | ~15 GB | ~5 GB (Subsampled @ 1fps)* |
| **FER2013** | 35k imágenes (48x48) | ~100 MB | ~100 MB |
| **NTHU-DDD** | Videos somnolencia | ~10 GB | ~2 GB (Subsampled) |
| **Modelos/Logs** | Checkpoints, .tflite, logs | < 1 GB | < 1 GB |
| **Entorno (.venv)** | Librerías Python | ~2 GB | - |
| **TOTAL** | | **~28-30 GB** | **~8-10 GB** |

> **(*) Estrategia de Ahorro:** No se recomienda extraer *todos* los frames de los videos (30 fps), ya que multiplicaría el peso por 10x. Para este proyecto, extraer **1 a 5 frames por segundo** es suficiente dada la naturaleza lenta de estados como "fatiga". O mejor aún, extraer solo **vectores de características** (arrays numéricos), lo que reduce GBs a MBs.

**🔴 Recomendación de Espacio Libre:** Se sugiere tener al menos **50 GB a 60 GB libres** en el disco duro para trabajar con holgura durante las etapas de descompresión y pruebas.

---

## 2. Requerimientos Computacionales (PC de Desarrollo)

Estos son los requisitos para la etapa de **Entrenamiento y Procesamiento** en tu computadora personal.

### A. Memoria RAM
*   **Mínimo:** 8 GB. (Puede requerir uso intensivo de archivo de paginación/swap al cargar videos).
*   **Recomendado:** **16 GB**. Permite cargar lotes de datos más grandes y correr el navegador/IDE simultáneamente sin lentitud.

### B. Procesador (CPU)
El preprocesamiento de video (OpenCV/MediaPipe) es intensivo en CPU.
*   **Mínimo:** Intel Core i5 (8va gen) o AMD Ryzen 5 (serie 3000) - 4 núcleos.
*   **Recomendado:** **6 núcleos o más**. Acelera drásticamente la extracción de landmarks faciales.

### C. Tarjeta Gráfica (GPU)
*   **Para Inferencia (Demo):** No es obligatoria. Los modelos MobileNet/Mini-Xception corren bien en CPU (aprox. 30-50ms por frame).
*   **Para Entrenamiento:**
    *   **Altamente Recomendada.** Entrenar modelos de video/imágenes en CPU puede tomar horas o días.
    *   **Sugerencia:** NVIDIA GTX 1650 (4GB VRAM) o superior.
    *   **Alternativa:** Si no tienes GPU dedicada, usa **Google Colab (Gratis)** para la etapa de entrenamiento (Notebook 03) y descarga el modelo `.tflite` para usarlo localmente.

---

## 3. Requerimientos para Despliegue (Edge Device - Objetivo Final)

Si decides probar esto en un dispositivo Edge (Raspberry Pi / Jetson) como indica la tesis:

*   **Raspberry Pi 4 / 5:** 4GB RAM mínimo. (Solo inferencia con TFLite).
*   **NVIDIA Jetson Nano:** 4GB RAM. (Ideal para este proyecto).

## Resumen Ejecutivo

Para trabajar localmente en tu PC ahora mismo:

1.  **Disco:** Libera **60 GB**.
2.  **RAM:** Con **16 GB** irás fluido. Con 8 GB, cierra otras apps pesadas.
3.  **Proceso:** Si tu PC no tiene GPU NVIDIA, **usa Google Colab para el entrenamiento** (Notebook 03) y tu PC para todo lo demás (Captura, Preprocesamiento, Demo).
