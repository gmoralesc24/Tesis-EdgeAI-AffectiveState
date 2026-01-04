# Verificación y Selección de Datasets
**Proyecto:** Medición de Estados Afectivos (Fatiga, Distracción, Atención, Otros)

Este documento analiza la validez de los datasets listados en `datasets_disponibles.csv` y su alineación con los 4 estados objetivo del proyecto.

## 1. Análisis de Alineación (Mapeo de Clases)

El proyecto requiere clasificar 4 estados. A continuación, se evalúa qué dataset cubre mejor cada estado:

| Estado Objetivo | Dataset Sugerido | Etiqueta Original / Mapeo | Nivel de Confianza |
| :--- | :--- | :--- | :--- |
| **1. Atención** | **DAiSEE** (Video) | `Engagement` (Nivel 3 y 4) | ⭐⭐⭐⭐⭐ (Alto) |
| | **ICCV 2021** (Video) | `Engaged` | ⭐⭐⭐⭐⭐ (Alto - Específico de aula) |
| **2. Distracción** | **DAiSEE** (Video) | `Engagement` (Nivel 0 y 1) | ⭐⭐⭐⭐ (Inferido por bajo engagement) |
| | **ICCV 2021** (Video) | `Wandering` (Mirando a otro lado) | ⭐⭐⭐⭐⭐ (Directo) |
| | **EngageNet** | `Disengaged` | ⭐⭐⭐⭐ |
| **3. Fatiga** | **DAiSEE** (Video) | `Boredom` (Aburrimiento Alto) | ⭐⭐⭐ (Aburrimiento $\approx$ Fatiga mental, pero no sueño físico) |
| | **NTHU-DDD** (Video)* | `Drowsy` (Bostezo, cabeceo) | ⭐⭐⭐⭐⭐ (Específico para fatiga/sueño) |
| **4. Otros** | **FER2013** (Img) | `Neutral`, `Surprise`, `Angry` | ⭐⭐⭐⭐ (Emociones base) |
| | **DAiSEE** | `Confusion`, `Frustration` | ⭐⭐⭐⭐ |

> (*) **Nota:** Se sugiere añadir **NTHU-DDD (Driver Drowsiness Detection)** si el objetivo "Fatiga" implica somnolencia física (cerrar ojos, cabecear), ya que DAiSEE solo mide "Aburrimiento".

## 2. Verificación de Datasets Disponibles

### A. DAiSEE (Dataset for Affective States in E-Environments) - **RECOMENDADO PRINCIPAL**
- **Estado:** ✅ **Válido y Activo**.
- **Tipo:** Video (Frames).
- **Justificación:** Es el estándar académico actual para "Atención" (Engagement) en entornos virtuales. Contiene 9000+ videos.
- **Acceso:** Público (requiere formulario simple).
- **Uso en Proyecto:** Fuente primaria para entrenar "Atención" vs "No Atención".

### B. FER2013 (Facial Expression Recognition) - **RECOMENDADO SECUNDARIO**
- **Estado:** ✅ **Válido y Activo (Kaggle)**.
- **Tipo:** Imágenes (48x48 px).
- **Justificación:** Excelente para pre-entrenar el modelo en reconocer rostros y expresiones básicas ("Otros").
- **Limitación:** Son imágenes estáticas, no capturan patrones temporales (como cabeceo por fatiga).
- **Uso en Proyecto:** Pre-entrenamiento (Transfer Learning).

### C. Datasets "Ideales" pero Difíciles de Obtener
- **DIPSER (2025):** Al ser muy reciente, es probable que no esté disponible públicamente aún. **Descartar para fase inicial**.
- **Student Engagement (ICCV):** Requiere contactar autores. Si responden, es la mejor opción para "Distracción".

## 3. Recomendación de Pipeline de Datos

Para cumplir con los objetivos sin depender de permisos complejos, sugiero este enfoque híbrido:

1.  **Descargar DAiSEE:** Usar como base para **Atención** y **Distracción** (bajo engagement).
2.  **Descargar FER2013:** Usar para la clase **Otros** (Neutral/Emociones) y para robustez visual.
3.  **Generación Propia (Script `capture.py`):**
    *   Grabar 10-20 videos simulando **Fatiga** (bostezos, ojos cerrados) y **Distracción** (mirar celular).
    *   Esto es crucial porque "Fatiga" real es difícil de encontrar en datasets de e-learning (donde la gente suele estar despierta).

## 4. Conclusión
Para las clases solicitadas:
- **Atención:** Usar DAiSEE.
- **Distracción:** Usar DAiSEE (Low Engagement) + Datos Propios.
- **Fatiga:** Generar datos propios (simulados) o buscar NTHU-DDD.
- **Otros:** Usar FER2013 (Neutral).

Este mix garantiza tener datos para validar el prototipo inmediatamente.
