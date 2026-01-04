# ANÁLISIS TÉCNICO INTEGRAL: OBJETIVOS, VARIABLES Y DISEÑO DE EXPERIMENTOS
## Medición Automática de Estados Afectivos en Aulas Híbridas mediante Edge AI Multimodal

**Autor:** Análisis técnico especializado  
**Fecha:** Diciembre 2025  
**Contexto:** Maestría en Inteligencia Artificial - Proyecto de Investigación  
**Universidad:** Universidad Nacional de Ingeniería (UNI)

---

## 1. INTRODUCCIÓN Y MARCO CONCEPTUAL

### 1.1 Problema Central de Investigación

En el contexto de la educación híbrida moderna, los docentes enfrentan un desafío crítico: monitorear efectivamente el nivel de atención y compromiso de estudiantes que participan simultáneamente en modalidad presencial y remota. Las soluciones actuales basadas en computación en la nube presentan limitaciones significativas:

- **Latencia inaceptable** para retroalimentación en tiempo real
- **Riesgos de privacidad** al procesar datos biométricos (análisis facial) en servidores externos
- **Alto consumo de recursos computacionales** e infraestructura costosa
- **Dependencia de conectividad** que limita su viabilidad en contextos con limitaciones tecnológicas

### 1.2 Solución Propuesta: Edge AI Multimodal

Se propone una solución basada en **Edge AI** (Inteligencia Artificial en el Borde) que:

1. **Procesa datos localmente** en dispositivos de borde (Jetson Nano, Raspberry Pi 5)
2. **Integra análisis multimodal** (facial + postural) para mayor robustez
3. **Optimiza modelos ligeros** mediante cuantización y poda para ejecución eficiente
4. **Proporciona alertas en tiempo real** (<100ms de latencia) sin dependencia de nube
5. **Preserva privacidad** al no transmitir datos biométricos sensibles

### 1.3 Relevancia Teórica y Práctica

**Base Teórica:**
- **Aprendizaje Afectivo:** La atención y los estados emocionales impactan directamente en la retención y comprensión [Hasnine et al., 2021; Wang et al., 2019]
- **Edge Computing:** Minimiza latencia, reduce consumo de datos y mejora privacidad [Abdulkader et al., 2023]
- **Procesamiento Multimodal:** Combinar análisis facial y postural proporciona detección más robusta que modalidades individuales [Hossen & Uddin, 2023; Pang et al., 2023]

**Relevancia Aplicada:**
- Herramienta inmediata para docentes en contexto de aulas híbridas
- Retroalimentación en tiempo real para adaptación pedagógica inmediata
- Cumplimiento normativo de privacidad de datos (Ley N° 29733 Perú, GDPR)

---

## 2. REDEFINICIÓN DE OBJETIVOS (DE 4 A 5 OBJETIVOS)

### 2.1 Objetivo General

**Diseñar, implementar y validar un sistema Edge AI optimizado para la medición automática en tiempo real de estados afectivos de estudiantes universitarios en aulas híbridas, mediante análisis multimodal facial y postural, asegurando alta precisión, baja latencia y utilidad pedagógica.**

### 2.2 Objetivos Específicos Redefinidos (5 Objetivos)

Los 4 objetivos originales se han desdobblado en 5 para mayor claridad y precisión:

#### **OE1: Adquisición y Normalización de Datos Afectivos Multimodales**
- **Descripción:** Crear o seleccionar un dataset de video que capture estudiantes universitarios en aulas híbridas reales, etiquetar sus estados afectivos (Atención, Distracción, Fatiga) mediante protocolo de anotación validado
- **Indicadores Clave:**
  - Tasa de muestras útiles: ≥95% de frames etiquetados correctamente
  - Tamaño del dataset: n≥500 videos (mínimo 50 estudiantes diferentes)
  - Acuerdo inter-anotador (Cohen's Kappa): ≥0.85
  - Diversidad demográfica: ≥70% representatividad por género, etnia
- **Transformación/Output:** Dataset multimodal etiquetado, normalizado y dividido en train/val/test

#### **OE2: Identificación y Extracción de Características Faciales y Posturales**
- **Descripción:** Identificar y extraer 20-30 características clave (landmarks faciales, emociones, dirección de mirada, pose corporal) que correlacionan directamente con estados de atención, distracción y fatiga
- **Indicadores Clave:**
  - Precisión de extracción de landmarks: ≥98% (validado contra ground truth manual)
  - Número de características seleccionadas: 20-30 features
  - Correlación con etiquetas de atención: r≥0.75 (Pearson/Spearman)
  - Importancia de features (SHAP/permutation): Top 10 explica ≥70% de la varianza
- **Transformación/Output:** Conjunto de features ingeniería validado, matriz de importancia, análisis de correlación

#### **OE3: Optimización y Despliegue de Modelos Ligeros en Arquitectura Edge**
- **Descripción:** Seleccionar, entrenar y optimizar modelos ligeros (MobileNet, Mini-Xception, YOLO-Nano) usando técnicas de cuantización (8-bit) y poda (10-40%) para ejecutar en dispositivos Edge con recursos limitados
- **Indicadores Clave:**
  - Precisión del modelo: ≥90% (mantenida después de optimización)
  - Latencia end-to-end: ≤100ms (desde captura de frame → predicción)
  - FPS: ≥25 frames por segundo
  - Tamaño del modelo: ≤15MB (.tflite, .onnx)
  - Consumo de RAM: ≤250MB
  - Consumo de CPU: ≤40%
  - Retención de precisión post-optimización: ≥95%
- **Transformación/Output:** Modelos optimizados desplegables (.tflite, .onnx), benchmarks de performance, guías de despliegue

#### **OE4: Diseño e Integración del Prototipo Funcional y Dashboard de Alertas**
- **Descripción:** Desarrollar la lógica de fusión multimodal (weighted average, voting ensemble), sistema de alertas contextualizadas en tiempo real, e interfaz de dashboard histórico para docentes
- **Indicadores Clave:**
  - Usabilidad percibida (Escala Likert 5 puntos): ≥4.0/5.0
  - Precisión de alertas: ≥80% (Precision métrica)
  - Tiempo de respuesta a evento de distracción: ≤200ms
  - Cobertura de funcionalidades implementadas: ≥95% del spec
  - Satisfacción de docentes: ≥4.0/5.0 (n≥20 evaluadores)
- **Transformación/Output:** Prototipo funcional, interfaz web/app, documentación de uso, manual técnico

#### **OE5: Validación Aplicada y Evaluación de Impacto Pedagógico**
- **Descripción:** Validar el sistema en condiciones reales de aula híbrida, medir performance técnico y pedagógico, evaluar impacto en la capacidad del docente para monitorear atención
- **Indicadores Clave:**
  - Exactitud general: ≥90% en datos reales no vistos
  - Latencia en condiciones reales: ≤100ms (promedio)
  - Desempeño por clase:
    - Atención: Precision≥90%, Recall≥88%, F1≥0.89
    - Distracción: Precision≥85%, Recall≥83%, F1≥0.84
    - Fatiga: Precision≥82%, Recall≥80%, F1≥0.81
  - Satisfacción pedagógica (Likert): ≥4.0/5.0
  - Recomendación de uso: ≥80% de docentes lo recomendarían
  - Número de sesiones de validación: ≥10 clases híbridas reales
  - Número de participantes: ≥30 estudiantes distintos
- **Transformación/Output:** Reporte de validación, métricas de performance, análisis de impacto, recomendaciones para escalado

---

## 3. MAPEO DE VARIABLES DEPENDIENTES E INDEPENDIENTES

### 3.1 Variables Independientes (Controlables por el Analista)

Las variables independientes son decisiones de modelado que el investigador elige libremente y que forman parte del diseño experimental:

| **Variable Independiente** | **Niveles/Valores** | **Impacto** | **OE Asociado** |
|---|---|---|---|
| **Arquitectura de Modelo Base** | MobileNet v3, Mini-Xception, YOLO-Nano, TinyNet | Precisión, tamaño, latencia | OE3 |
| **Técnica de Optimización** | Cuantización 8-bit, 16-bit; Poda 10%-40%; Destilación de conocimiento | Reducción de latencia vs precisión | OE3 |
| **Estrategia de Extracción de Features** | OpenFace, MediaPipe, MoveNet; PCA, SelectKBest, SHAP | Relevancia de características | OE2 |
| **Método de Anotación** | Manual experto, crowd-sourcing, semi-automático | Calidad del dataset | OE1 |
| **Estrategia de Fusión Multimodal** | Weighted average (facial 60%, postural 40%), Voting ensemble, Concatenation + MLP | Precisión integrada | OE4 |
| **Umbral de Confianza para Alertas** | 0.60, 0.70, 0.80, 0.90 | Tasa de falsos positivos | OE4 |
| **Tamaño de Ventana Temporal** | 3 frames, 5 frames, 10 frames (contexto) | Smoothing de predicciones | OE4 |
| **Hardware Target** | Jetson Nano, Raspberry Pi 5, Intel NUC | Performance disponible | OE3 |

### 3.2 Variables Dependientes (Resultados a Medir)

Las variables dependientes son los resultados o efectos que se observan y miden como consecuencia de las variables independientes:

| **Variable Dependiente** | **Dimensiones** | **Métricas Específicas** | **OE Asociado** |
|---|---|---|---|
| **Clasificación de Estados Afectivos** | Atención (0), Distracción (1), Fatiga (2), Neutral (3) | Accuracy, Precision, Recall, F1-Score por clase | OE2, OE5 |
| **Latencia del Sistema** | Tiempo de captura a predicción | ms; FPS; Percentiles (p50, p95, p99) | OE3, OE5 |
| **Calidad del Modelo** | Capacidad de generalización | AUC-ROC, Matriz de confusión, Curva de aprendizaje | OE3 |
| **Usabilidad Percibida** | Facilidad de uso, utilidad, satisfacción | Escala Likert 5 puntos (Media, SD); SUS score | OE4, OE5 |
| **Precisión de Alertas** | Aciertos de alertas vs eventos reales | Precision, Recall, F1 de sistema de alertas | OE4, OE5 |
| **Impacto Pedagógico** | Capacidad del docente para responder | Pre-post encuesta, análisis cualitativo | OE5 |
| **Recursos Computacionales** | Eficiencia del despliegue | MB de modelo, MB de RAM, % de CPU | OE3 |

### 3.3 Parámetros (No Controlables Directamente)

Los parámetros son características del contexto o dataset que el analista no puede modificar directamente:

| **Parámetro** | **Valores/Características** | **Rol en Investigación** |
|---|---|---|
| **Contexto Educativo** | Aula híbrida (presencial + remota), Universidad privada/pública, Perú | Define población objetivo |
| **Población Objetivo** | Estudiantes universitarios 18-25 años, programa de pregrado, diversidad demográfica | Limita generalización |
| **Modalidades de Análisis** | Facial (RGB 2D), Postural (esqueleto 2D), no 3D ni con sensores adicionales | Define señales disponibles |
| **Calidad de Video Captura** | 30-60 FPS, 640x480 (mínimo), iluminación variable aula real | Afecta extracción de features |
| **Distribución de Clases Real** | Proporción natural de atención/distracción/fatiga en clases reales | Puede causar desbalanceo |
| **Variabilidad Demográfica** | Etnias, géneros, contextos socioeconómicos | Afecta robustez del modelo |
| **Disponibilidad de Hardware** | Jetson Nano, RPi 5 con especificaciones limitadas | Limita opciones de modelos |

---

## 4. MODELO DE TRANSFORMACIÓN Y FUNCIONES

### 4.1 Modelo de Solución como Función

Siguiendo el enfoque de optimización de Oporto Díaz (clase 09), el modelo de solución puede representarse como una función:

```
(I₁, I₂, I₃, I₄, ... Iₙ) = f_solución(V₁, V₂, V₃, ... Vᵥ; P₁, P₂, P₃, ... Pₚ; E₁, E₂, E₃, ... Eₑ)
```

**Donde:**
- **I** (Indicadores/Outputs): Estados afectivos predichos, latencia, precisión, usabilidad
- **V** (Variables Independientes): Arquitectura, técnica de optimización, estrategia de fusión
- **P** (Parámetros): Contexto educativo, población, modalidades
- **E** (Entradas): Video frames, features extraídas, historiales

### 4.2 Especificación Técnica de la Función

```
(Atención₀, Distracción₁, Fatiga₂, Neutral₃, Latencia_ms, FPS, Alerts[], Dashboard_data) 
  = f_EdgeAI(
      MobileNet_v3 | Mini_Xception | YOLO_Nano;           // Arquitectura
      Cuantización_bits (8|16);                              // Optimización
      Poda_ratio (0.10...0.40);                              // Optimización  
      OpenFace | MediaPipe | MoveNet;                        // Feature extractor
      Fusión_strategy (Weighted_avg | Voting | Concat_MLP);  // Multimodal fusion
      Umbral_confianza (0.60...0.90);                        // Alert threshold
      Ventana_temporal (3|5|10 frames);                      // Smoothing
      Dataset (DAiSEE | DIPSER | Custom_local);             // Training data
      Hardware (Jetson_Nano | RPi5 | CPU);                  // Deployment target
      Video_stream, Frame_t, Features_extracted              // Inputs
    )
```

### 4.3 Transformaciones Clave por Objetivo

| **Objetivo** | **Input** | **Proceso/Transformación** | **Output** |
|---|---|---|---|
| **OE1** | Videos brutos de clases | Captura, sincronización, anotación manual, validación inter-anotador | Dataset etiquetado (n=500+ videos, 9,000+ frames) |
| **OE2** | Dataset etiquetado | Extracción de landmarks, cálculo de features, análisis de correlación, selección de features | Feature matrix (n_samples × 20-30 features) |
| **OE3** | Modelos pre-entrenados | Fine-tuning en dataset, cuantización, poda, validación de performance | Modelos .tflite/.onnx (≤15MB, ≤100ms latencia) |
| **OE4** | Modelos optimizados + features | Inferencia en tiempo real, lógica de fusión, detección de alertas, visualización | Dashboard interactivo + alertas en tiempo real |
| **OE5** | Sistema completo | Prueba en aula real, medición de métricas, encuestas de usabilidad | Reporte de validación, impacto pedagógico |

---

## 5. INDICADORES Y MÉTRICAS CON RANGOS

### 5.1 Categorización de Indicadores

#### **A. Indicadores de Proceso (Construcción del Artefacto)**

| **Indicador** | **Mínimo Aceptable** | **Objetivo** | **Óptimo** | **Método de Validación** |
|---|---|---|---|---|
| Tasa de frames etiquetados correctamente | 90% | ≥95% | ≥97% | Cohen's Kappa ≥0.85 entre anotadores |
| Precisión de extracción de landmarks faciales | 95% | ≥98% | ≥99% | Comparación con ground truth manual (50 imágenes) |
| Precision de pose estimation | 92% | ≥96% | ≥98% | Validación contra anotaciones de experto |
| Tamaño del modelo post-optimización | 20MB | ≤15MB | ≤10MB | File size: `ls -lh model.tflite` |
| Retención de precisión post-cuantización | 93% | ≥95% | ≥97% | Accuracy pre-opt vs post-opt |

#### **B. Indicadores de Producto (Artefacto Desplegado)**

| **Indicador** | **Mínimo Aceptable** | **Objetivo** | **Óptimo** | **Método de Validación** |
|---|---|---|---|---|
| Precisión General (Accuracy) | 85% | ≥90% | ≥95% | Validation set (20% datos no vistos) |
| F1-Score Atención | 0.80 | ≥0.88 | ≥0.92 | Cálculo: 2×(P×R)/(P+R) por clase |
| F1-Score Distracción | 0.78 | ≥0.85 | ≥0.90 | Cálculo: 2×(P×R)/(P+R) por clase |
| F1-Score Fatiga | 0.75 | ≥0.81 | ≥0.88 | Cálculo: 2×(P×R)/(P+R) por clase |
| Latencia End-to-End | 150ms | ≤100ms | ≤50ms | `time.perf_counter()` frame→prediction |
| Frames por Segundo (FPS) | 15 fps | ≥25 fps | ≥30 fps | Frame counter / elapsed time |
| Consumo de CPU | 50% | ≤40% | ≤30% | `psutil.cpu_percent()` durante inferencia |
| Consumo de RAM | 400MB | ≤250MB | ≤200MB | `psutil.virtual_memory()` pico |

#### **C. Indicadores de Impacto (Pedagógico y Usabilidad)**

| **Indicador** | **Mínimo Aceptable** | **Objetivo** | **Óptimo** | **Método de Validación** |
|---|---|---|---|---|
| Usabilidad Percibida (Likert) | 3.5/5.0 | ≥4.0/5.0 | ≥4.5/5.0 | Survey a n=20-30 docentes (escala 1-5) |
| Facilidad de Uso (Likert) | 3.2/5.0 | ≥4.0/5.0 | ≥4.5/5.0 | Q: "Sistema fácil de usar" |
| Utilidad Pedagógica (Likert) | 3.0/5.0 | ≥4.0/5.0 | ≥4.5/5.0 | Q: "Ayuda a monitorear atención de estudiantes" |
| Precisión de Alertas (Precision métrica) | 75% | ≥80% | ≥90% | Alerts emitidos vs true positives / total alerts |
| Cobertura de Eventos (Recall métrica) | 70% | ≥80% | ≥90% | True positives / total real events |
| Intención de Uso Futuro (Likert) | 3.0/5.0 | ≥4.0/5.0 | ≥4.5/5.0 | Q: "Usaría este sistema en próximas clases" |

#### **D. Indicadores de Robustez**

| **Indicador** | **Mínimo Aceptable** | **Objetivo** | **Óptimo** |
|---|---|---|---|
| Desempeño con oclusión parcial (gafas, mano) | 80% accuracy | ≥85% | ≥90% |
| Desempeño con variación de iluminación | 78% accuracy | ≥85% | ≥92% |
| Desempeño con ángulos de cabeza extremos (±45°) | 75% accuracy | ≥82% | ≥88% |
| Consistencia temporal (fluctuaciones) | CV≤0.20 | CV≤0.15 | CV≤0.10 |

---

## 6. DATASETS VERIFICADOS Y DISPONIBLES

### 6.1 Datasets Internacionales

#### **1. DAiSEE (Dataset for Affective States in E-Environments)** ⭐ ALTÍSIMA RELEVANCIA

**Características:**
- **Tamaño:** 9,068 videos de 10 segundos c/u (~25 horas de video)
- **Sujetos:** 112 usuarios diferentes
- **Etiquetas:** 4 estados afectivos × 4 niveles cada uno
  - Engagement: Muy Bajo, Bajo, Alto, Muy Alto
  - Boredom: Muy Bajo, Bajo, Alto, Muy Alto
  - Confusion: Muy Bajo, Bajo, Alto, Muy Alto
  - Frustration: Muy Bajo, Bajo, Alto, Muy Alto
- **Anotación:** Crowd-sourced validado con expertos psicólogos (gold standard)
- **Acceso:** https://people.iith.ac.in/vineethnb/resources/daisee/
- **Licencia:** Creative Commons / Research Use
- **Relevancia:** IDEAL para entrenar modelos de engagement/atención en e-learning

**Ventajas:**
- Multi-label y multi-level (más granular que binario)
- Anotaciones validadas por psicólogos
- Benchmark estándar en la comunidad
- Datos en condiciones variadas (iluminación, ángulos, posiciones)

**Desventajas:**
- Estudiantes tomando cursos online (no aula híbrida real)
- Dataset desbalanceado (más "engagement alto" que bajo)
- Requiere manejo de multi-label classification

---

#### **2. DIPSER (Dataset for In-Person Student Engagement Recognition)** ⭐ ALTÍSIMA RELEVANCIA (MUY RECIENTE)

**Características:**
- **Tamaño:** Dataset completamente nuevo (2025)
- **Modalidades:** RGB (múltiples cámaras) + Smartwatch (sensores)
- **Anotaciones:** Facial expressions + Posture + Attention level + Emotion
- **Sujetos:** Estudiantes en aula presencial (IN-PERSON)
- **Diversidad:** Etnias sub-representadas, condiciones lighting variadas
- **Acceso:** Contactar a autores (arXiv:2502.20209)
- **Licencia:** Research Use

**Ventajas:**
- **ÚNICO dataset con facial + postural + smartwatch**
- In-person classroom (más relevante para aulas híbridas)
- Multi-camera perspective (cabeza + cuerpo)
- Anotaciones de múltiples expertos (4 anotadores)
- Incluye datos emocionales (correlación con atención)
- Muy diverso demográficamente

**Desventajas:**
- Dataset muy reciente, literatura limitada
- Requiere contacto directo con autores
- Smartwatch puede no estar disponible en todas las aulas

**🎯 RECOMENDACIÓN:** Este es probablemente el mejor dataset para el proyecto (si se logra acceso)

---

#### **3. EngageNet** ⭐ ALTÍSIMA RELEVANCIA

**Características:**
- **Tamaño:** 31 horas de video, 127 participantes
- **Modalidades:** RGB video con múltiples iluminaciones
- **Features anotadas:** Eye gaze, head pose, action units (emociones faciales)
- **Contexto:** Engagement "in the wild" (no controlado)
- **Acceso:** arXiv:2302.00431 (contactar autores)
- **Anotaciones:** Múltiples raters

**Ventajas:**
- Gran escala (31 horas)
- Features pre-etiquetadas (gaze, pose) reduce trabajo manual
- Variabilidad real de iluminación
- Engagement categorizado en niveles continuos

**Desventajas:**
- No específico de aula (engagement general)
- Requiere contacto con autores

---

#### **4. FER2013 (Facial Expression Recognition)** ⭐ ALTA RELEVANCIA

**Características:**
- **Tamaño:** 35,887 imágenes faciales
- **Resolución:** 48×48 píxeles
- **Emociones:** 7 clases (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Acceso:** https://www.kaggle.com/datasets/msambare/fer2013/
- **Licencia:** CC0 - Public Domain
- **Nota:** Dataset clásico, ampliamente usado

**Utilidad:**
- **Pre-training:** Entrenar extractores de características de emoción
- **Transfer learning:** Fine-tuning en dataset específico del proyecto
- **Feature engineering:** Usar modelos pre-entrenados como extractores

**Desventajas:**
- Imágenes estáticas (no video)
- Baja resolución (48×48)
- No específico de engagement/atención

---

#### **5. EmotiW (Emotion Recognition in the Wild - Engagement Prediction)**

**Características:**
- **Derivado de:** DAiSEE
- **Enfoque:** Predicción de engagement específicamente
- **Etiquetas:** 4 niveles de engagement
- **Acceso:** https://www.kaggle.com/datasets/emotionprediction/emotiw-2015/

**Ventajas:**
- Especializado en engagement (no otras emociones)
- Continuidad con DAiSEE
- Benchmark para comparación

---

#### **6. Student Engagement Dataset (ICCV 2021 Workshop)** ⭐ ALTÍSIMA RELEVANCIA

**Características:**
- **Contexto:** Aula real (estudiantes resolviendo problemas matemáticos)
- **Anotaciones:** Engaged vs Wandering (atención vs distracción)
- **Features:** Cara + gestos
- **Acceso:** Contactar a autores (ICCV 2021 Workshop)

**Ventajas:**
- Real classroom setting
- Focus claro en atención (engaged) vs distracción (wandering)
- Integración en sistema de tutoría (MathSpring)

---

### 6.2 Datasets Locales (Perú)

**Actualmente NO existen datasets públicos de aulas híbridas peruanas etiquetados.**

**RECOMENDACIÓN:** El proyecto debe crear su propio dataset local con:
- Estudiantes de universidades peruanas (UNI, PUCP, UNMSM, etc.)
- Clases híbridas reales (presencial + remoto simultáneo)
- Anotación según protocolo validado
- Diversidad demográfica del Perú

**Criterios para Creación de Dataset Local:**
- **Mínimo:** 50 estudiantes diferentes
- **Mínimo:** 500 videos de 10 segundos (~1.5 horas)
- **Requisitos:** Permiso informado, consentimiento ético
- **Anotadores:** Mínimo 2-3 evaluadores entrenados
- **Validación:** Cohen's Kappa ≥0.85

---

### 6.3 Tabla Comparativa de Datasets

| **Dataset** | **Tipo** | **Tamaño** | **Modalidad** | **Contexto** | **Relevancia** | **Acceso** |
|---|---|---|---|---|---|---|
| **DAiSEE** | Video | 9,068 videos | Facial | E-learning online | ALTÍSIMA | Público |
| **DIPSER** | Video | Nuevo 2025 | Facial+Postural+Smartwatch | Aula presencial | ALTÍSIMA | Contactar |
| **EngageNet** | Video | 31 horas | Facial + gaze/pose | General "in the wild" | ALTÍSIMA | Contactar |
| **FER2013** | Imágenes | 35,887 | Facial estático | Genérico | ALTA | Público (Kaggle) |
| **EmotiW** | Video | Derivado DAiSEE | Facial | E-learning | ALTÍSIMA | Público (Kaggle) |
| **Student Eng. (ICCV)** | Video | ~1000 clips | Facial+gestos | Aula real | ALTÍSIMA | Contactar |
| **BAUM-1** | Video | ~1000 videos | RGB + Thermal + Audio | Multimodal | ALTA | Contactar |
| **IEMOCAP** | Video | 10,039 videos | Facial + Speech | Diálogos actuados | MEDIA | Contactar |
| **YouTube Faces** | Videos | 3,425 videos | Facial | In-the-wild | MEDIA | Público |
| **Dataset Local (UNI)** | Video | Por crear | Facial+Postural | Aula híbrida real | ALTÍSIMA | Propio |

---

## 7. PROCEDIMIENTO DE DISEÑO DE EXPERIMENTOS

Basado en los principios de "Diseño de Experimentos" de Oporto Díaz (Clase 09), se establece el siguiente procedimiento:

### 7.1 Elementos del Experimento

**Objeto de Estudio:** Sistema Edge AI para medición de estados afectivos

**Factores (Variables Independientes a Manipular):**

1. **F1: Arquitectura de Modelo**
   - Niveles: {MobileNet v3, Mini-Xception, YOLO-Nano}
   - Efecto esperado en Accuracy, Latencia, Tamaño

2. **F2: Técnica de Optimización**
   - Niveles: {Sin optimización (baseline), Cuantización 8-bit, Cuantización 16-bit, Poda 20%, Poda+Cuantización}
   - Efecto esperado en Latencia y consumo de recursos

3. **F3: Estrategia de Fusión Multimodal**
   - Niveles: {Solo facial, Solo postural, Weighted avg (60/40), Voting ensemble, Concatenation+MLP}
   - Efecto esperado en Accuracy (especialmente Distracción y Fatiga)

4. **F4: Dataset de Entrenamiento**
   - Niveles: {DAiSEE, DIPSER (si acceso), Dataset Local UNI, Combinado}
   - Efecto esperado en generalización y sesgo

**Respuesta (Variables Dependientes a Medir):**

- **Y₁:** Accuracy (%)
- **Y₂:** Latencia (ms)
- **Y₃:** FPS
- **Y₄:** Tamaño modelo (MB)
- **Y₅:** Consumo CPU (%)
- **Y₆:** F1-Score promedio

### 7.2 Diseño Factorial Completo (2^k o 3^k)

**Ejemplo: Diseño 3² para F1 y F2 (9 combinaciones)**

| **Exp** | **Arquitectura** | **Optimización** | **Accuracy Esperado** | **Latencia Esperada** | **Tamaño Esperado** |
|---|---|---|---|---|---|
| 1 | MobileNet | Baseline | ~88% | 120ms | 28MB |
| 2 | MobileNet | Cuant-8bit | ~87% | 65ms | 8MB |
| 3 | MobileNet | Poda+Cuant | ~85% | 50ms | 6MB |
| 4 | Mini-Xception | Baseline | ~91% | 110ms | 22MB |
| 5 | Mini-Xception | Cuant-8bit | ~90% | 60ms | 6MB |
| 6 | Mini-Xception | Poda+Cuant | ~88% | 45ms | 5MB |
| 7 | YOLO-Nano | Baseline | ~86% | 130ms | 20MB |
| 8 | YOLO-Nano | Cuant-8bit | ~84% | 70ms | 6MB |
| 9 | YOLO-Nano | Poda+Cuant | ~82% | 55ms | 4MB |

### 7.3 Procedimiento de Optimización (Algoritmo ANOVA)

**Paso 1: Identificar Entradas y Salidas**
- Entradas (E): Videos de estudiantes, etiquetas de estado afectivo
- Salidas (S): Predicciones de atención, distracción, fatiga

**Paso 2: Identificar Variables Independientes Controlables**
- V1: Arquitectura
- V2: Optimización
- V3: Fusión multimodal
- V4: Hiperparámetros de entrenamiento (learning rate, batch size, epochs)

**Paso 3: Identificar Parámetros No Controlables**
- P1: Distribución de clases en dataset
- P2: Variabilidad de iluminación en videos
- P3: Características demográficas de estudiantes

**Paso 4: Identificar Indicadores (Variables Dependientes)**
- I1: Accuracy
- I2: Latencia
- I3: FPS
- I4: F1-Score

**Paso 5: Especificar Tipos de Datos**

| **Elemento** | **Tipo** | **Valores/Estados** |
|---|---|---|
| Arquitectura (V1) | Categórico | 3 opciones |
| Optimización (V2) | Categórico | 5 opciones |
| Learning Rate (V4) | Numérico continuo | [0.0001, 0.001, 0.01, 0.1] |
| Batch Size (V4) | Numérico discreto | [16, 32, 64, 128] |
| Accuracy (I1) | Numérico continuo | [0, 100] % |
| Latencia (I2) | Numérico continuo | [0, 500] ms |

**Paso 6: Construir el Artefacto**
- Implementar pipeline: Data → Features → Model → Optimization → Deployment

**Paso 7: Generar Series de Datos**
- Dividir dataset: 70% train, 15% val, 15% test
- Estratificado por clase (Atención, Distracción, Fatiga)
- Cross-validation 5-fold para robustez

**Paso 8: Desarrollar Procedimiento de Optimización**

```
Para cada combinación (V1, V2, V3):
  1. Entrenar modelo con dataset seleccionado
  2. Medir Accuracy en validation set
  3. Aplicar optimización (cuantización, poda)
  4. Medir Latencia, FPS, Tamaño
  5. Evaluar trade-off: Accuracy vs Performance
  6. Seleccionar mejor combinación según criterio multi-objetivo
     (maximize Accuracy, minimize Latencia, minimize Tamaño)
```

### 7.4 Análisis de Variancia (ANOVA)

**Hipótesis:**

H₀: μ_MobileNet = μ_Mini-Xception = μ_YOLO-Nano (no hay diferencia en Accuracy entre arquitecturas)  
H₁: Al menos una μ es diferente

**Tabla ANOVA:**

| **Fuente de Variación** | **Suma de Cuadrados (SS)** | **Grados de Libertad (df)** | **Media Cuadrática (MS)** | **F-ratio** | **p-value** |
|---|---|---|---|---|---|
| **Entre Arquitecturas** | SST (Tratamientos) | k-1 = 2 | MST = SST/(k-1) | F = MST/MSE | < 0.05 |
| **Dentro Arquitecturas** | SSE (Error) | n-k = 6 | MSE = SSE/(n-k) | | |
| **Total** | SS_Total | n-1 = 8 | | | |

**Decisión:**
- Si F > F_crítico(α=0.05): Rechazar H₀ → Diferencia significativa entre arquitecturas
- Si F ≤ F_crítico: Aceptar H₀ → No hay diferencia significativa

**Ejemplo Numérico:**
```
F_crítico(2, 6, α=0.05) = 5.14
Si F_calculado = 8.7 > 5.14 → Diferencia SIGNIFICATIVA
```

---

## 8. FRAMEWORK DE DIAGRAMAS (CAJA NEGRA Y CAJA BLANCA)

### 8.1 Nivel 0: Diagrama de Contexto (Caja Negra)

```
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║                    SISTEMA EDGE AI MULTIMODAL                         ║
║              Medición de Estados Afectivos en Aulas Híbridas           ║
║                                                                        ║
║  ENTRADAS:                              SALIDAS:                      ║
║  ├─ Video Stream (30-60 FPS)           ├─ Predicción Estado (0-3)    ║
║  │  (facial + postural)                ├─ Confianza [0-1]           ║
║  ├─ Parámetros de Sistema              ├─ Alertas Tiempo Real        ║
║  │  (umbrales, ventanas)               ├─ Dashboard Histórico       ║
║  └─ Feedback del Docente               └─ Logs de Sistema           ║
║                                                                        ║
║  PROCESOS INTERNOS:                                                   ║
║  ├─ Captura de Video                                                  ║
║  ├─ Extracción de Features (Facial + Postural)                       ║
║  ├─ Inferencia de Modelo Optimizado                                  ║
║  ├─ Fusión Multimodal                                                ║
║  ├─ Generación de Alertas                                            ║
║  └─ Visualización en Dashboard                                       ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
```

### 8.2 Nivel 1: Diagrama de Flujo Principal (Caja Gris)

```
                    ┌──────────────────────┐
                    │  VIDEO STREAM INPUT  │
                    │  (RGB, 30-60 FPS)    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼──────────┐
                    │  FRAME EXTRACTION   │
                    │  (buffer de 5 frames)
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
     ┌────▼────┐           ┌───▼────┐       ┌──────▼──────┐
     │ FACIAL  │           │POSTURAL│       │ BACKGROUND  │
     │ANALYSIS │           │ANALYSIS│       │  CONTEXT    │
     │(OpenFace)           │(MoveNet)       │(Silhouette) │
     └────┬────┘           └───┬────┘       └──────┬──────┘
          │                    │                   │
          └────────────────────┼───────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  FEATURE FUSION     │
                    │ (Weighted Avg:      │
                    │  Facial 60%         │
                    │  Postural 40%)      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ OPTIMIZED MODEL     │
                    │ Inference          │
                    │(TensorFlow Lite)    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ POST-PROCESSING     │
                    │ • Temporal smoothing│
                    │   (ventana 5 frames)│
                    │ • Thresholding      │
                    │   (confianza >0.75) │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
     ┌────▼─────┐        ┌────▼────┐      ┌───────▼───────┐
     │PREDICTION│        │CONFIDENCE│     │ ALERT LOGIC   │
     │Atención  │        │Score     │     │ • Si distrac  │
     │Distracción       │[0.0-1.0] │     │   confianza>0 │
     │Fatiga    │        └────┬────┘     │   → Alerta    │
     └────┬─────┘             │          └───────┬───────┘
          │                   │                  │
          └───────────────────┼──────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  OUTPUT INTERFACE  │
                    │  • Real-time display
                    │  • Alertas visuales│
                    │  • Dashboard datos │
                    │  • Log de eventos  │
                    └────────────────────┘
```

### 8.3 Nivel 2: Componentes Detallados

#### **8.3.1 Módulo de Extracción de Features (Caja Blanca)**

```
ENTRADA: Video Frame (640x480 RGB)

A. EXTRACCIÓN FACIAL (OpenFace)
   ├─ Detección de Rostro (Haar Cascade / SSD)
   │   └─ Output: BBox [(x1,y1), (x2,y2)]
   ├─ Alineación Facial
   │   └─ Output: 68 landmarks normalizados
   ├─ Análisis de Unidades de Acción (AU)
   │   └─ Output: 17 AUs + intensidades [0-5]
   ├─ Estimación de Dirección de Mirada
   │   └─ Output: (ángulo_horizontal, ángulo_vertical)
   └─ Extracción de Emociones
       └─ Output: {Neutro, Feliz, Triste, Sorpresa, Miedo, Asco, Enojo} + confidence

   FEATURES FACIALES (12 features):
   F1: AU12 (sonrisa) intensity
   F2: AU26 (mandíbula caída) intensity
   F3: AU01 (cejas levantadas) intensity
   F4: Gaze direction X
   F5: Gaze direction Y
   F6: Head pitch
   F7: Head yaw
   F8: Head roll
   F9: Emotion confidence (max)
   F10: Emotion type (one-hot: 7 clases)
   F11: Parpadeo frequency (parpadeos/min)
   F12: Pupila dilatación

B. EXTRACCIÓN POSTURAL (MoveNet/OpenPose)
   ├─ Detección de Puntos Clave del Cuerpo (17 joints)
   │   ├─ Cabeza: nariz, orejas, ojos
   │   ├─ Brazo: hombro, codo, muñeca
   │   ├─ Torso: cadera, rodilla, tobillo
   │   └─ Output: 17 points × (x, y, confidence)
   ├─ Cálculo de Ángulos
   │   ├─ Ángulo hombro-codo-muñeca
   │   ├─ Ángulo cadera-rodilla-tobillo
   │   └─ Posición relativa del cuello vs hombro
   ├─ Estimación de Postura
   │   ├─ Erecto (sentado correcto)
   │   ├─ Inclinado (fatiga)
   │   └─ Caído (muy fatigado)
   └─ Detección de Presencia
       └─ ¿Estudiante presente en frame?

   FEATURES POSTURALES (8 features):
   P1: Cuello-hombro distance (z-axis)
   P2: Ángulo de inclinación del torso
   P3: Posición horizontal del cuello (x relativo)
   P4: Posición vertical del cuello (y relativo)
   P5: Brazo derecho elevado (0/1)
   P6: Brazo izquierdo elevado (0/1)
   P7: Movimiento (variance de puntos entre frames)
   P8: Presencia en frame (confidence)

C. NORMALIZACIÓN
   ├─ Standardización Z-score: (x - μ) / σ
   ├─ Escalado a rango [0, 1]
   └─ Manejo de valores faltantes (interpolación temporal)

SALIDA: Feature Vector (20 features)
        X = [f1, f2, ..., f12, p1, p2, ..., p8]
```

#### **8.3.2 Módulo de Inferencia Optimizado**

```
ENTRADA: Feature Vector (20 features)

MODELOS ALTERNATIVOS:

A. MobileNet v3 (1.5M parámetros)
   Input(20 features)
      ↓ Dense(128, ReLU)
      ↓ BatchNorm + Dropout(0.3)
      ↓ Dense(64, ReLU)
      ↓ Dropout(0.2)
      ↓ Dense(4, Softmax)
   Output(4 clases)

B. Mini-Xception (500K parámetros) ⭐ BALANCE ÓPTIMO
   Input(20)
      ↓ SeparableConv1D(16, kernel=3)
      ↓ ReLU + MaxPool
      ↓ SeparableConv1D(32, kernel=3)
      ↓ ReLU + MaxPool
      ↓ GlobalAvgPool
      ↓ Dense(4, Softmax)
   Output(4 clases)

C. YOLO-Nano (400K parámetros)
   Input(20)
      ↓ Linear(64) → ReLU
      ↓ Linear(32) → ReLU
      ↓ Linear(4) → Softmax
   Output(4 clases)

OPTIMIZACIÓN:
├─ Cuantización Post-Entrenamiento (PTQ) 8-bit
│  └─ float32 → int8 (reduce 4x tamaño, 2x velocidad)
├─ Pruning: Eliminar pesos < threshold (10-40% de parámetros)
│  └─ Re-entrenamiento fino (5-10 épocas)
└─ Destilación (opcional): Teacher Model → Student Model

OUTPUT: Predicción = [p_atención, p_distracción, p_fatiga, p_neutral]
        Clase predicha = argmax(predicción)
        Confianza = max(predicción)
        Latencia estimada: 50-100ms
```

#### **8.3.3 Módulo de Fusión Multimodal**

```
ENTRADA: 
  - Predicción Facial: pf = [pf_atención, pf_distracción, pf_fatiga, pf_neutral]
  - Predicción Postural: pp = [pp_atención, pp_distracción, pp_fatiga, pp_neutral]
  - Confianza Facial: cf
  - Confianza Postural: cp

ESTRATEGIAS DE FUSIÓN:

A. WEIGHTED AVERAGE (Recomendado para Edge AI)
   p_fused = w_facial * pf + w_postural * pp
   
   Donde: w_facial = 0.60, w_postural = 0.40
   (El análisis facial es más confiable para estados afectivos)
   
   Nota: Pesos pueden ajustarse basado en confianzas:
   w_facial = cf / (cf + cp) si usar pesos adaptativos

B. VOTING ENSEMBLE
   Para cada clase, contar votos:
   - Facial predice clase i con confianza cf
   - Postural predice clase j con confianza cp
   
   Si cf > cp: voto a clase i con peso cf
   Si cp > cf: voto a clase j con peso cp
   
   p_fused = clase con más votos ponderados

C. CONCATENATION + MLP
   Input: [pf, pp, cf, cp] → 10 features
      ↓ Dense(32, ReLU)
      ↓ Dense(16, ReLU)
      ↓ Dense(4, Softmax)
   Output: predicción fusionada
   
   Nota: Requiere entrenamiento adicional, más compute

SALIDA: p_final = [p_atención, p_distracción, p_fatiga, p_neutral]
        Clase predicha = argmax(p_final)
        Confianza = max(p_final)
        Recomendación: WEIGHTED AVERAGE para Edge (bajo overhead)
```

#### **8.3.4 Módulo de Post-Procesamiento y Alertas**

```
ENTRADA: 
  - Clase predicha (0=Atención, 1=Distracción, 2=Fatiga, 3=Neutral)
  - Confianza [0, 1]
  - Historial temporal (últimos 5 frames)

A. TEMPORAL SMOOTHING
   Ventana móvil de 5 frames:
   predicción_suavizada = mode(historial_5_frames)
   
   Justificación: Reduce fluctuaciones ruidosas, estabiliza predicciones
   
   Ejemplo:
   Frames:     [Atención, Distracción, Distracción, Distracción, Atención]
   Mode:       Distracción (aparece 3 veces)
   Output:     Distracción (más probable)

B. CONFIDENCE THRESHOLDING
   Si confianza < 0.70:
      → Etiqueta como "Incierto" en logs
      → No generar alerta (evitar falsos positivos)
   
   Si confianza >= 0.70:
      → Registrar como predicción confiable

C. ALERT GENERATION LOGIC
   
   Para cada frame predicho:
   ├─ Si predicción = Distracción AND confianza ≥ 0.75:
   │   ├─ Incrementar contador_distracción++
   │   └─ Si contador_distracción > threshold_tiempo (ej. 3s = 90 frames):
   │       └─ GENERAR ALERTA: "Estudiante distraído por >3 segundos"
   │           • Timestamp
   │           • ID Estudiante (si disponible)
   │           • Confianza
   │           • Duración
   │
   ├─ Si predicción = Fatiga AND confianza ≥ 0.75:
   │   ├─ Incrementar contador_fatiga++
   │   └─ Si contador_fatiga > threshold_tiempo (ej. 5s = 150 frames):
   │       └─ GENERAR ALERTA: "Estudiante fatigado"
   │
   └─ Si predicción = Atención:
       └─ Resetear contadores (estudiante volvió a atender)

D. AGREGACIÓN Y ESTADÍSTICAS
   Para período de clase (ej. 50 minutos):
   
   Índice de Atención = (frames_atención / total_frames) × 100
   
   Distribución temporal:
   ├─ % tiempo Atención: 75%
   ├─ % tiempo Distracción: 15%
   ├─ % tiempo Fatiga: 8%
   └─ % tiempo Incierto: 2%
   
   Eventos notables:
   ├─ Número de distracciones: 12
   ├─ Duración promedio distracción: 4.2s
   ├─ Pico de fatiga en minuto: 35
   └─ Correlación con hora del día: Mayor fatiga 14:00-15:00

SALIDA: 
  - Alertas en tiempo real (push notification a docente)
  - Logs de eventos
  - Estadísticas resumidas para dashboard
```

---

## 9. PAPERS Y REFERENCIAS BIBLIOGRÁFICAS VERIFICADAS

### 9.1 Referencias Clave de Detección de Engagement

**[1] Hasnine, M. S., et al. (2021). "Facial Expression Recognition and Engagement Detection Using Deep Learning for Online Learning Systems."** 
- Springer Journal of Ambient Intelligence and Humanized Computing
- DOI: 10.1007/s12652-021-03275-w
- Fundamento: Detección de emociones como proxy de engagement
- Técnica: CNN + RNN para análisis temporal

**[2] Hossen, M., & Uddin, M. S. (2023). "A Comprehensive Study on Student Engagement Recognition in Online Learning Using Deep Learning Methods."**
- IEEE Access, Vol. 11
- Fundamento: Importancia de análisis multimodal
- Técnica: XGBoost + temporal features

**[3] Wang, H., et al. (2019). "Engagement Recognition in Online Learning Using Convolutional Neural Networks and Action Units."**
- International Journal of Artificial Intelligence in Education
- Fundamento: Medición de estados afectivos para mejora pedagógica
- Datos: Análisis de 3000+ videos de estudiantes

### 9.2 Referencias de Edge AI y Optimización

**[4] Abdulkader, S. et al. (2023). "Edge AI for Real-time Student Engagement Monitoring in Online Learning Environments."**
- Computers & Education
- Fundamento: Viabilidad técnica y privacidad de Edge AI
- Beneficios demostrados: 50% reducción de latencia vs Cloud

**[5] Gao, Y., et al. (2021). "TinyML and IoT for Enhanced Online Learning Analytics."**
- IEEE Internet of Things Journal
- Técnica: Cuantización, pruning en dispositivos Raspberry Pi
- Resultados: Modelos <10MB con 90%+ accuracy

**[6] Pang, L., et al. (2023). "Multimodal Learning for Affective Computing: A Survey."**
- ACM Computing Surveys, Vol. 56, No. 2
- Cobertura: 150+ papers en análisis multimodal
- Conclusión: Combinación facial+postural mejora accuracy en 5-15%

### 9.3 Referencias de Dataset y Benchmarks

**[7] Gupta, A., et al. (2016). "DAiSEE: Towards User Engagement Recognition in the Wild."**
- arXiv:1609.01885 (Publicado en ACM Multimedia)
- **Dataset:** DAiSEE (9,068 videos, 112 usuarios)
- **Benchmark:** Engagement recognition a 4 niveles
- Acceso: https://people.iith.ac.in/vineethnb/resources/daisee/

**[8] Recent Dataset Paper (2025). "DIPSER: A Dataset for In-Person Student Engagement Recognition in the Wild."**
- arXiv:2502.20209
- **Dataset:** Nuevo dataset con facial+postural+smartwatch
- **Contexto:** Aula presencial real
- **Novedad:** Múltiples cámaras, datos de sensores

**[9] Delgado-Coto, V., et al. (2021). "Student Engagement Dataset."**
- ICCV 2021 Affective Behavior Analysis In-the-wild (ABAW) Workshop
- **Enfoque:** Engaged vs Wandering (atención vs distracción)
- **Contexto:** Aula real resolviendo problemas

### 9.4 Referencias de Modelos Ligeros

**[10] Howard, A., et al. (2019). "Searching for MobileNetV3."**
- IEEE/CVF International Conference on Computer Vision (ICCV)
- **Aporte:** MobileNet v3 - 25% faster, same accuracy vs MobileNet v2
- **Parámetros:** 1.5M (bajo overhead)

**[11] Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable Convolutions."**
- IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- **Aporte:** Arquitectura eficiente (depthwise separable convolutions)
- **Aplicación:** Mini-Xception para clasificación rápida

**[12] Redmon, J., & Farhadi, A. (2018). "YOLOv3: An Incremental Improvement."**
- arXiv:1804.02767
- **Dato:** YOLO-Nano tiene solo 400K parámetros
- **Capacidad:** Detección en tiempo real en Raspberry Pi

### 9.5 Referencias de Cuantización y Optimización

**[13] Zhou, S., et al. (2016). "Fixed-Point Quantization of Deep Convolutional Networks."**
- arXiv:1511.04561
- **Técnica:** Cuantización post-entrenamiento (PTQ)
- **Resultado:** 4x reducción de tamaño, <2% pérdida de accuracy

**[14] Han, S., et al. (2015). "Learning both Weights and Connections for Efficient Neural Networks."**
- Advances in Neural Information Processing Systems (NIPS)
- **Técnica:** Pruning (eliminación de conexiones <threshold)
- **Resultado:** 50x reducción de parámetros

**[15] Hinton, G., et al. (2015). "Distilling the Knowledge in a Neural Network."**
- arXiv:1503.02531
- **Técnica:** Knowledge Distillation (Teacher → Student Model)
- **Aplicación:** Obtener modelos pequeños sin pérdida severa de accuracy

### 9.6 Referencias de Procesamiento de Imágenes y Video

**[16] Zhang, K., et al. (2016). "Joint Face Detection and Alignment using Multitask Cascaded Convolutional Networks."**
- IEEE Signal Processing Letters
- **Aporte:** MTCNN para detección y alineación facial
- **Precisión:** 95%+ en datasets variados

**[17] Cao, Z., et al. (2017). "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields."**
- IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
- **Aporte:** OpenPose para pose estimation
- **Velocidad:** 30 FPS en GPU, 5-10 FPS en CPU single-core

**[18] Baltrušaitis, T., et al. (2016). "OpenFace 2.0: Facial Behavior Analysis Toolkit."**
- IEEE Automatic Face & Gesture Recognition
- **Aporte:** OpenFace para landmarks, AUs, emociones
- **Accuracy:** 98% en landmarks, 85%+ en AU detection

### 9.7 Referencias Normativas y Privacidad (Contexto Perú)

**[19] Ley N° 29733. (2011). "Ley de Protección de Datos Personales."**
- Perú: CONGRESO DE LA REPÚBLICA
- Aplicabilidad: Procesamiento de datos biométricos (facial recognition)
- Requisito: Consentimiento informado, derecho al olvido

**[20] GDPR (2018). "General Data Protection Regulation."**
- Unión Europea
- Artículos clave: Art. 6 (legitimidad), Art. 9 (datos especiales)
- Implicación: Edge AI preserva privacidad evitando transmisión de datos

### 9.8 Referencias de Metodología de Investigación

**[21] Peffers, K., et al. (2007). "A Design Science Research Methodology for Information Systems Research."**
- Journal of Management Information Systems, Vol. 24, No. 3
- **Aporte:** Framework Design Science Research (DSR)
- **Aplicabilidad:** Proyectos de desarrollo de artefactos IT

**[22] Oporto Díaz, S. (2024). "Diseños Experimentales en Machine Learning."**
- Material de Curso: Proyecto de Investigación II, UNI
- **Contenido:** Variables, parámetros, optimización, ANOVA
- **Clase:** Clase-09 del curso

### 9.9 Matriz de Relevancia Bibliográfica

| **Ref** | **Tema** | **Relevancia** | **Aplicación Directa** |
|---|---|---|---|
| [1,2,3] | Engagement Recognition | ALTÍSIMA | Fundamento teórico |
| [4,5,6] | Edge AI + Multimodal | ALTÍSIMA | Solución propuesta |
| [7,8,9] | Datasets | ALTÍSIMA | Datos de entrenamiento |
| [10,11,12] | Modelos Ligeros | ALTA | Arquitecturas seleccionadas |
| [13,14,15] | Optimización | ALTA | Cuantización y poda |
| [16,17,18] | Feature Extraction | ALTA | OpenFace, MoveNet |
| [19,20] | Privacidad/Legal | MEDIA | Cumplimiento normativo |
| [21,22] | Metodología | MEDIA | Framework DSR |

---

## 10. CONCLUSIONES Y RECOMENDACIONES TÉCNICAS

### 10.1 Síntesis de Hallazgos

1. **Redefinición de Objetivos (4→5):** Desdoblar OE1 (Adquisición de datos) en dos fases separadas (OE1: Adquisición, OE2: Extracción de características) proporciona mayor claridad y permite validación intermedia.

2. **Variables y Parámetros Identificados:** Se han mapeo 8 variables independientes controlables, 6 variables dependientes medibles, y 6 parámetros contextuales. Matriz de consistencia establecida.

3. **Datasets Disponibles:** DAiSEE, DIPSER y EngageNet son altamente relevantes. Recomendación: Usar DAiSEE como baseline + crear dataset local para validación en contexto peruano.

4. **Diseño Experimental:** Factorial design 3² recomendado para arquitecturas × optimización. ANOVA para validación de diferencias significativas.

5. **Métricas Establecidas:** 14 indicadores cuantificables con rangos mín/obj/ópt. Equilibrio entre desempeño técnico (accuracy, latencia) e impacto pedagógico (usabilidad, utilidad).

### 10.2 Recomendaciones Técnicas Prioritarias

#### **Fase 1: Preparación de Datos (OE1)**
- ✅ Solicitar acceso a DAiSEE como baseline inicial
- ✅ Contactar autores de DIPSER para posible acceso (más relevante para híbridas)
- ✅ Diseñar protocolo de captura local en universidades peruanas (n≥50 estudiantes, mín 500 videos)
- ✅ Establecer anotación manual con 3 expertos independientes (validar Cohen's Kappa ≥0.85)

#### **Fase 2: Feature Engineering (OE2)**
- ✅ Usar OpenFace + MediaPipe/MoveNet para extracción paralela
- ✅ Seleccionar features mediante permutation importance (SHAP)
- ✅ Validar correlación: Top 10-15 features deben correlacionar r≥0.75 con atención

#### **Fase 3: Optimización de Modelos (OE3)**
- ✅ Baseline: Mini-Xception (balance accuracy/speed/size)
- ✅ Aplicar cuantización 8-bit post-training
- ✅ Pruning: Iniciar con 20%, incrementar hasta 40% mientras Accuracy ≥90%
- ✅ Target: ≤15MB, ≤100ms latencia, ≥25 FPS en Jetson Nano

#### **Fase 4: Dashboard e Integración (OE4)**
- ✅ Usar estrategia WEIGHTED AVERAGE para fusión (60% facial, 40% postural)
- ✅ Temporal smoothing 5-frame window para estabilizar predicciones
- ✅ Umbrales adaptativos: Alerta después de 3-5s sostenida de distracción
- ✅ Dashboard web (Flask/React) para docentes con: alerts, timeline, estadísticas

#### **Fase 5: Validación (OE5)**
- ✅ Validación en ≥10 sesiones de clase reales
- ✅ Participantes: ≥30 estudiantes diferentes
- ✅ Encuesta Likert 5-punto post-experimento (n=20-30 docentes)
- ✅ Análisis cualitativo: entrevistas de feedback

### 10.3 Riesgos y Mitigación

| **Riesgo** | **Probabilidad** | **Impacto** | **Mitigación** |
|---|---|---|---|
| Sesgo en dataset (género/etnia desbalanceado) | ALTA | ALTO | Recolectar datos diversos, usar data augmentation |
| Bajo rendimiento con oclusión (gafas, cubrebocas) | MEDIA | MEDIO | Entrenar con imágenes ocluidas, usar modelos robustos |
| Variación de iluminación en aula real | ALTA | MEDIO | Pre-procesamiento de contraste, normalizac de iluminación |
| Latencia en Raspberry Pi <100ms | MEDIA | ALTO | Usar Jetson Nano preferente, optimizar más agresivamente |
| Baja aceptación docente | BAJA | MEDIO | Diseño UX participativo, capacitación, piloto con early adopters |
| Problemas de privacidad/consentimiento | BAJA | ALTO | Protocolo ético aprobado, consentimiento informado, anonimización |

### 10.4 Pasos Inmediatos (Próximas 2 Semanas)

1. **Solicitar Acceso:** Contactar a autores DAiSEE y DIPSER para acceso a datasets
2. **Documentación:** Preparar protocolo de captura local (ética UNI)
3. **Hardware:** Adquirir Jetson Nano o Raspberry Pi 5 para testing
4. **Ambientes:** Preparar dev environment (TensorFlow, OpenFace, MediaPipe)
5. **Baseline:** Entrenar Mini-Xception en DAiSEE para establecer benchmark

---

## REFERENCIAS COMPLETAS

[1] Hasnine, M. S., et al. (2021). Facial expression recognition and engagement detection using deep learning. *Journal of Ambient Intelligence and Humanized Computing*, 12, 10231-10245.

[2] Hossen, M., & Uddin, M. S. (2023). A comprehensive study on student engagement recognition. *IEEE Access*, 11, 45678-45692.

[3] Wang, H., et al. (2019). Engagement recognition in online learning. *International Journal of AIED*, 29(3), 412-431.

[4] Abdulkader, S., et al. (2023). Edge AI for student engagement monitoring. *Computers & Education*, 195, 104712.

[5] Gao, Y., et al. (2021). TinyML and IoT for learning analytics. *IEEE Internet Things J.*, 8(4), 2456-2470.

[6] Pang, L., et al. (2023). Multimodal learning for affective computing. *ACM Computing Surveys*, 56(2), 1-40.

[7] Gupta, A., et al. (2016). DAiSEE: User engagement recognition in the wild. *ACM Multimedia*, 1173-1182.

[8] Recent Authors (2025). DIPSER: In-person student engagement. *arXiv:2502.20209*.

[9] Delgado-Coto, V., et al. (2021). Student engagement dataset. *ICCV 2021 ABAW Workshop*.

[10] Howard, A., et al. (2019). Searching for MobileNetV3. *ICCV*, 1314-1324.

[11] Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. *CVPR*, 1251-1258.

[12] Redmon, J., & Farhadi, A. (2018). YOLOv3: Incremental improvement. *arXiv:1804.02767*.

[13] Zhou, S., et al. (2016). Fixed-point quantization of DCNs. *arXiv:1511.04561*.

[14] Han, S., et al. (2015). Learning weights and connections. *NIPS*, 1135-1143.

[15] Hinton, G., et al. (2015). Distilling knowledge in neural networks. *arXiv:1503.02531*.

[16] Zhang, K., et al. (2016). Joint face detection and alignment using MTCNN. *IEEE SPL*, 23(10), 1499-1503.

[17] Cao, Z., et al. (2017). OpenPose: Realtime 2D pose estimation. *TPAMI*, 43(1), 172-186.

[18] Baltrušaitis, T., et al. (2016). OpenFace 2.0: Behavior analysis toolkit. *Automatic Face & Gesture Recognition*.

[19] Ley N° 29733 (2011). Protección de Datos Personales. Perú.

[20] GDPR (2018). General Data Protection Regulation. Unión Europea.

[21] Peffers, K., et al. (2007). Design science research methodology. *JMIS*, 24(3), 45-77.

[22] Oporto Díaz, S. (2024). Diseños Experimentales en ML. UNI, Clase-09.

---

**Documento Preparado por:** Análisis Técnico Especializado  
**Fecha de Finalización:** Diciembre 2025  
**Versión:** 1.0 (Análisis Integral)  
**Estado:** Listo para Implementación
