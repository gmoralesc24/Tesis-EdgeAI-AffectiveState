# Estrategia Edge AI y Optimización
**Proyecto:** Medición Automática de Estados Afectivos en Aulas Híbridas

Este documento detalla cómo el proyecto se alinea con los principios de **Edge AI** (Inteligencia Artificial en el Borde), cumpliendo con los requisitos de baja latencia, privacidad y autonomía, fundamentales para el Objetivo Específico 3 (OE3).

## 1. Justificación del Enfoque Edge AI
En el contexto de un aula híbrida, procesar video en la nube (Cloud AI) presenta problemas críticos:
- **Latencia:** Enviar video HD a un servidor y esperar respuesta introduce retrasos >500ms, inútiles para alertas en tiempo real.
- **Privacidad:** La transmisión de rostros de estudiantes fuera del aula viola normativas de protección de datos (ej. Ley 29733 en Perú).
- **Ancho de Banda:** Streaming continuo de múltiples cámaras satura la red de la universidad.

**Solución Edge:** El procesamiento ocurre *in-situ* (en el dispositivo del aula), eliminando la transferencia de video.

## 2. Selección de Hardware Objetivo
El sistema está diseñado para ejecutarse en dispositivos de bajo consumo (SBCs):
- **NVIDIA Jetson Nano:** (GPU 128 cores) - Ideal para modelos acelerados por CUDA.
- **Raspberry Pi 4/5:** (CPU ARM) - Económico, requiere modelos altamente optimizados (TFLite).

## 3. Pipeline de Optimización (OE3)
Para lograr inferencia en estos dispositivos con FPS > 25, aplicamos:

### 3.1 Arquitecturas Eficientes
En lugar de modelos pesados (ResNet, VGG), utilizamos:
- **MobileNetV3:** Diseñado con "depthwise separable convolutions" que reducen operaciones en 8x.
- **Mini-Xception:** Versión reducida de Xception, ideal para 48x48 píxeles (emociones).

### 3.2 Cuantización (Quantization)
Convertimos los pesos del modelo de precisión flotante (FP32) a enteros (INT8).
- **Beneficio:** Reducción de tamaño del modelo (4x menor, de 20MB a 5MB).
- **Aceleración:** Las operaciones INT8 son más rápidas en CPUs ARM y DSPs.
- **Herramienta:** TensorFlow Lite Converter (`src/models/optimize.py`).

### 3.3 Poda (Pruning)
Eliminación de conexiones neuronales redundantes (pesos cercanos a cero) para reducir el cómputo sin perder precisión significativa.

## 4. Workflows de Desarrollo e Implementación
1. **Entrenamiento (GPU Workstation/Colab):** Se entrena el modelo "Maestro" en FP32.
2. **Optimización:** Convertir a `.tflite` (INT8).
3. **Despliegue:** Copiar `.tflite` al dispositivo Edge.
4. **Inferencia:** Ejecutar con `src/app/inference.py` usando `tflite_runtime` (librería ligera).

## 5. métricas de Éxito para Edge
Se considera exitoso si:
- **Latencia:** < 100ms (tiempo real).
- **Tamaño:** < 50MB (para caber en RAM limitada de microcontroladores/RPi).
- **Privacidad:** Ningún frame de video sale del dispositivo; solo se envían metadatos (ej. "Atención: Alta").
