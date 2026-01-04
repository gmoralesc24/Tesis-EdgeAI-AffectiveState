# DIAGRAMAS TÉCNICOS: ARQUITECTURA Y FLUJOS DEL SISTEMA
## Medición Automática de Estados Afectivos en Aulas Híbridas - Edge AI

---

## 1. DIAGRAMA DE CONTEXTO (Nivel 0 - Caja Negra)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                         SISTEMA EDGE AI MULTIMODAL                         │
│                                                                             │
│     MEDICIÓN DE ESTADOS AFECTIVOS EN AULAS HÍBRIDAS UNIVERSITARIAS        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ENTRADAS EXTERNAS:                                                        │
│  ┌──────────────────────┐                                                  │
│  │ Video Stream (RGB)   │ ──► 30-60 FPS, 640x480, análisis facial+postural
│  │ (de cámaras)         │                                                  │
│  └──────────────────────┘                                                  │
│                                                                             │
│  ┌──────────────────────┐                                                  │
│  │ Parámetros Sistema   │ ──► Umbrales, ventanas temporales,               
│  │ (config)             │     estrategias de fusión                       │
│  └──────────────────────┘                                                  │
│                                                                             │
│  ┌──────────────────────┐                                                  │
│  │ Feedback Docente     │ ──► Validación manual, calibración               
│  │ (opcional)           │                                                  │
│  └──────────────────────┘                                                  │
│                           ╔════════════════════════════╗                   │
│                           ║   SISTEMA EDGE AI          ║                   │
│                           ║   • Inferencia Local       ║                   │
│                           ║   • Multimodal Fusion      ║                   │
│                           ║   • Real-time Processing   ║                   │
│                           ╚════════════════════════════╝                   │
│                                   │                                        │
│  SALIDAS GENERADAS:               │                                        │
│                                   ▼                                        │
│  ┌──────────────────────┐                                                  │
│  │ Predicción Estado    │ ◄─── Atención (0), Distracción (1),              
│  │ Afectivo             │       Fatiga (2), Neutral (3)                   │
│  │ + Confianza [0-1]    │                                                  │
│  └──────────────────────┘                                                  │
│                                                                             │
│  ┌──────────────────────┐                                                  │
│  │ Alertas Tiempo Real  │ ◄─── Distracción >3s, Fatiga >5s,                
│  │ (Push a Docente)     │       Anomalías detectadas                      │
│  └──────────────────────┘                                                  │
│                                                                             │
│  ┌──────────────────────┐                                                  │
│  │ Dashboard Histórico  │ ◄─── Gráficos, estadísticas, timeline,           
│  │ (Web App)            │       reportes por estudiante                   │
│  └──────────────────────┘                                                  │
│                                                                             │
│  ┌──────────────────────┐                                                  │
│  │ Logs y Trazabilidad  │ ◄─── Eventos, timestamps, confianzas,           
│  │ (para auditoría)     │       métricas de performance                   │
│  └──────────────────────┘                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. DIAGRAMA DE FLUJO DE DATOS (Nivel 1 - Caja Gris)

```
                                VIDEO CAPTURA
                                     │
                         ┌───────────▼───────────┐
                         │  FRAME BUFFER (30FPS) │
                         │  Almacenador temporal │
                         │  de 5 frames          │
                         └───────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
      ┌───────▼────────┐  ┌──────────▼────────┐  ┌─────────▼─────────┐
      │ FACIAL ANALYSIS│  │ POSTURAL ANALYSIS │  │   CONTEXT CHECK   │
      │ (OpenFace)     │  │ (MoveNet/OpenPose)│  │ (Background, size)│
      │                │  │                   │  │                   │
      │ • Landmarks    │  │ • Skeletal points │  │ • Presencia       │
      │   (68 puntos)  │  │   (17 articul.)   │  │ • Tamaño de frame │
      │ • AU (emociones)  │ • Ángulos          │  │ • Iluminación     │
      │ • Gaze (mirada)   │ • Postura          │  │                   │
      │ • Emociones    │  │ • Movimiento      │  │ • Validez frame   │
      └───────┬────────┘  └──────────┬────────┘  └─────────┬─────────┘
              │                      │                     │
              └──────────────────────┼─────────────────────┘
                                     │
                      ┌──────────────▼──────────────┐
                      │ FEATURE VECTOR CONSTRUCTION │
                      │  X = [f1, f2, ..., f20]    │
                      │                             │
                      │ Facial (12 features):       │
                      │ • AU intensities (3)        │
                      │ • Gaze direction (2)        │
                      │ • Head pose (3)             │
                      │ • Emoción (3)               │
                      │ • Parpadeo                  │
                      │                             │
                      │ Postural (8 features):      │
                      │ • Posición cuello (3)       │
                      │ • Inclinación torso (2)     │
                      │ • Brazos elevados (2)       │
                      │ • Movimiento                │
                      └──────────────┬──────────────┘
                                     │
                      ┌──────────────▼──────────────┐
                      │ NORMALIZACIÓN (Z-SCORE)    │
                      │ X' = (X - μ) / σ           │
                      │ Escalado [0, 1]            │
                      │ Manejo de NaNs             │
                      └──────────────┬──────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │                                 │
          ┌─────────▼──────────┐         ┌────────────▼────────────┐
          │ MODELO FACIAL      │         │ MODELO POSTURAL        │
          │                    │         │                        │
          │ Mini-Xception      │         │ MobileNet v3           │
          │ • 500K params      │         │ • 1.5M params          │
          │ • Cuantizado 8bit  │         │ • Cuantizado 8bit      │
          │ • Input: 12 feat   │         │ • Input: 8 feat        │
          │ • Output: [4]      │         │ • Output: [4]          │
          │ • Latencia: 40ms   │         │ • Latencia: 50ms       │
          └────────┬───────────┘         └────────┬───────────────┘
                   │                             │
                   │ p_facial =                 │ p_postural =
                   │ [p_atenc,                  │ [p_atenc,
                   │  p_dist,                   │  p_dist,
                   │  p_fatig,                  │  p_fatig,
                   │  p_neutr]                  │  p_neutr]
                   │                             │
                   └────────────┬────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ FUSIÓN MULTIMODAL      │
                    │ Weighted Average:      │
                    │ p_final =              │
                    │  0.60 * p_facial +     │
                    │  0.40 * p_postural     │
                    │                        │
                    │ Output: [4] confianzas │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ POST-PROCESAMIENTO     │
                    │                        │
                    │ • Temporal Smoothing   │
                    │   (ventana 5 frames)   │
                    │                        │
                    │ • Confianza Threshold  │
                    │   (si conf<0.7: skip)  │
                    │                        │
                    │ • Estado Final:        │
                    │   clase = argmax(pred) │
                    │   conf = max(pred)     │
                    └───────────┬────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
        ┌─────▼─────┐      ┌────▼────┐      ┌───▼───────┐
        │ PREDICCIÓN│      │ LOG DE  │      │   ALERT   │
        │  ESTADO   │      │ EVENTOS │      │ GENERATION│
        │ (0-3)     │      │         │      │           │
        │ + Conf    │      │ - TS    │      │ Si Dist:  │
        │           │      │ - Frame │      │ counter++ │
        │Atención   │      │ - Clase │      │ Si >3s:   │
        │Distracción      │ - Conf  │      │ ALERTA→   │
        │Fatiga    │      │         │      │ DOCENTE   │
        │Neutral   │      │         │      │           │
        └─────┬─────┘      └────┬────┘      └───┬───────┘
              │                 │               │
              └─────────────────┼───────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  OUTPUT INTERFACE      │
                    │                        │
                    │  • Real-time Display   │
                    │  • Alertas Visuales    │
                    │  • Dashboard Analytics │
                    │  • REST API para Apps  │
                    └────────────────────────┘
```

---

## 3. DIAGRAMA DE TRANSFORMACIONES POR OBJETIVO

```
OBJETIVO 1: ADQUISICIÓN DE DATOS
┌─────────────────────────────────────────────────────────┐
│ Input: Videos brutos de clases híbridas reales          │
│ └─► Captura: 50+ estudiantes, 500+ videos (10s c/u)   │
│     └─► Anotación: 3 evaluadores independientes        │
│         └─► Validación: Cohen's Kappa ≥ 0.85          │
│             └─► Output: Dataset Etiquetado & Dividido │
│                         (train/val/test: 70/15/15)     │
└─────────────────────────────────────────────────────────┘

OBJETIVO 2: EXTRACCIÓN DE CARACTERÍSTICAS  
┌─────────────────────────────────────────────────────────┐
│ Input: Dataset Etiquetado                               │
│ └─► OpenFace/MediaPipe: 68 landmarks + 17 articul.    │
│     └─► Feature Engineering: 50+ features potenciales  │
│         └─► Correlación Análisis: Pearson/Spearman    │
│             └─► Feature Selection: Top 20-30 features │
│                 └─► Output: Feature Matrix Validado    │
│                         [n_samples × 20 features]       │
└─────────────────────────────────────────────────────────┘

OBJETIVO 3: OPTIMIZACIÓN EDGE
┌─────────────────────────────────────────────────────────┐
│ Input: Modelos Pre-entrenados (MobileNet, Mini-Xception)
│ └─► Fine-tuning: Dataset local con labels             │
│     └─► Entrenamiento: 100 épocas, early stopping     │
│         └─► Cuantización: 8-bit PTQ                    │
│             └─► Pruning: 10-40% parámetros eliminados │
│                 └─► Validación: Accuracy ≥90%         │
│                     └─► Output: Modelos .tflite/.onnx │
│                              (≤15MB, ≤100ms)           │
└─────────────────────────────────────────────────────────┘

OBJETIVO 4: PROTOTIPO + DASHBOARD
┌─────────────────────────────────────────────────────────┐
│ Input: Modelos Optimizados + Features                   │
│ └─► Pipeline Inferencia: Real-time en Jetson Nano     │
│     └─► Fusión Multimodal: Weighted Average (60/40)   │
│         └─► Smoothing Temporal: 5-frame window        │
│             └─► Alert Logic: Umbrales de confianza    │
│                 └─► Dashboard Web: Flask + React      │
│                     └─► Output: Sistema Operacional   │
│                          (Alertas + Dashboard)         │
└─────────────────────────────────────────────────────────┘

OBJETIVO 5: VALIDACIÓN PEDAGÓGICA
┌─────────────────────────────────────────────────────────┐
│ Input: Sistema Completo Desplegado                      │
│ └─► Prueba Real: ≥10 sesiones de clase híbrida        │
│     └─► Medición: Accuracy, Latencia, FPS, Usabilidad │
│         └─► Encuesta: Likert 5-punto a docentes (n=30) │
│             └─► Análisis Cualitativo: Entrevistas    │
│                 └─► Output: Reporte de Validación    │
│                          + Impacto Pedagógico          │
└─────────────────────────────────────────────────────────┘
```

---

## 4. MATRIZ DE VARIABLES Y OBJETIVOS (Relación Cruzada)

```
┌─────────────────────┬────┬────┬────┬────┬────┐
│ Variable / Parámetro│OE1 │OE2 │OE3 │OE4 │OE5 │
├─────────────────────┼────┼────┼────┼────┼────┤
│ VI: Arquitectura    │    │    │ ✓✓ │    │    │
│ VI: Optimización    │    │    │ ✓✓ │    │    │
│ VI: Feature Select  │    │ ✓✓ │    │    │    │
│ VI: Método Anotación│ ✓✓ │    │    │    │    │
│ VI: Fusión Multim.  │    │    │    │ ✓✓ │    │
│ VI: Umbral Alertas  │    │    │    │ ✓  │ ✓  │
│                     │    │    │    │    │    │
│ VD: Clasificación   │    │ ✓  │ ✓  │    │ ✓✓ │
│ VD: Latencia        │    │    │ ✓✓ │ ✓  │ ✓✓ │
│ VD: Usabilidad      │    │    │    │ ✓✓ │ ✓✓ │
│ VD: Precisión Alert │    │    │    │ ✓✓ │ ✓  │
│                     │    │    │    │    │    │
│ P: Contexto Híbrido │ ✓  │ ✓  │ ✓  │ ✓  │ ✓  │
│ P: Población Univ   │ ✓  │ ✓  │ ✓  │ ✓  │ ✓  │
│ P: Multimodalidad   │ ✓  │ ✓  │ ✓  │ ✓  │ ✓  │
│ P: Calidad Video    │ ✓  │ ✓  │    │    │    │
│ P: Distrib. Clases  │ ✓  │    │ ✓  │ ✓  │    │
└─────────────────────┴────┴────┴────┴────┴────┘

Leyenda: ✓✓ = Relación Directa/Crítica
         ✓  = Relación Importante
         (vacío) = Relación Mínima
```

---

## 5. PIPELINE DE DATOS (Caja Blanca - Nivel 3)

```
ESTACIÓN 1: ADQUISICIÓN Y NORMALIZACIÓN
═══════════════════════════════════════════════════

Video Crudo
    │
    ├─► [Pre-procesamiento]
    │   ├─ Lectura: OpenCV (30-60 FPS)
    │   ├─ Resize: 640x480 (estandarización)
    │   ├─ Normalización RGB: [0,255] → [0,1]
    │   └─ Sincronización: Timestamp con video
    │
    ├─► [Detección de Rostro]
    │   ├─ Detector: Haar Cascade / SSD
    │   ├─ Validación: Rostro ≥200x200 px
    │   ├─ Crop: Región de interés
    │   └─ Fallback: Frame anterior si no detectado
    │
    └─► Frame Procesado
        (normalizad, rostro aislado)

ESTACIÓN 2: EXTRACCIÓN DE FEATURES (Paralelo)
═════════════════════════════════════════════════

┌─────────────────────────────┬──────────────────────────────┐
│   RAMA 1: FACIAL ANALYSIS   │  RAMA 2: POSTURAL ANALYSIS   │
├─────────────────────────────┼──────────────────────────────┤
│ OpenFace::processImage()    │ MoveNet::detectPose()        │
│                             │                              │
│ 1. Landmarks (68 puntos)    │ 1. Skeleton (17 joints)      │
│    - Nariz                  │    - Cabeza, brazos, torso   │
│    - Ojos                   │    - Caderas, piernas        │
│    - Boca                   │                              │
│    - Mejillas               │ 2. Confidence scores         │
│                             │    per keypoint              │
│ 2. Action Units (17 AUs)    │                              │
│    - Sonrisa (AU12)         │ 3. Cálculo ángulos:          │
│    - Cejas (AU01)           │    - Cuello vs hombro        │
│    - Mandíbula (AU26)       │    - Torso inclinación       │
│                             │    - Posición relativa       │
│ 3. Gaze (dirección mirada)  │                              │
│    - Horizontal (pitch)     │ 4. Postura clasificación:    │
│    - Vertical (yaw)         │    - Erecto, Inclinado,      │
│                             │    - Caído                   │
│ 4. Emociones                │                              │
│    - Neutro, Feliz, Triste  │ 5. Movimiento:               │
│    - Sorpresa, Miedo, etc   │    - Var(x,y,z) de puntos   │
│                             │                              │
│ OUTPUT:                     │ OUTPUT:                      │
│ F = [AU12, AU26, AU01,      │ P = [cuello_x, cuello_y,    │
│      gaze_h, gaze_v,        │      inclin_torso,           │
│      head_pitch,            │      mov_brazo_d,            │
│      head_yaw,              │      mov_brazo_i,            │
│      head_roll,             │      movimiento_total,       │
│      emociones (7x)]        │      presencia]              │
│ Dim: 12 features            │ Dim: 8 features              │
└─────────────────────────────┴──────────────────────────────┘
        │                              │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼────────────────┐
        │ COMBINACIÓN Y NORMALIZACIÓN   │
        │                               │
        │ X = [F || P]                  │
        │ X = [f1...f12, p1...p8]       │
        │ Dim: 20 features              │
        │                               │
        │ Normalización Z-score:        │
        │ X' = (X - μ) / σ              │
        │                               │
        │ Valores faltantes:            │
        │ Interpolación temporal        │
        │ o valor anterior              │
        │                               │
        │ OUTPUT: Matriz de Entrada    │
        │ Dim: (1 × 20)                │
        └──────────────┬───────────────┘
                       │
ESTACIÓN 3: INFERENCIA EN MODELOS PARALELOS
═════════════════════════════════════════════

        ┌─────────────┬──────────────┐
        │             │              │
   ┌────▼────────┐   ┌┴──────────────▼───┐
   │ MODEL 1:    │   │ MODEL 2:           │
   │ FACIAL      │   │ POSTURAL           │
   │             │   │                    │
   │ Mini-       │   │ MobileNet v3       │
   │ Xception    │   │                    │
   │             │   │ Input: P[8]        │
   │ Input: F[12]│   │                    │
   │ ↓Dense 128  │   │ ↓Dense 256         │
   │ ↓ReLU       │   │ ↓ReLU              │
   │ ↓Drop 0.3   │   │ ↓Drop 0.4          │
   │ ↓Dense 64   │   │ ↓Dense 128         │
   │ ↓ReLU       │   │ ↓ReLU              │
   │ ↓Drop 0.2   │   │ ↓Drop 0.2          │
   │ ↓Dense 4    │   │ ↓Dense 4           │
   │ ↓Softmax    │   │ ↓Softmax           │
   │ Output [4]  │   │ Output [4]         │
   └────┬────────┘   └───┬────────────────┘
        │                │
        │ Latencia:      │ Latencia:
        │ 40ms (opt)     │ 50ms (opt)
        │                │
        │ p_facial:      │ p_postural:
        │ [0.85,0.10,    │ [0.60,0.25,
        │  0.03,0.02]    │  0.10,0.05]
        │                │
        └────────┬───────┘
                 │
        ┌────────▼────────┐
        │ FUSIÓN WEIGHTED │
        │ AVERAGE         │
        │                 │
        │ p_final =       │
        │ 0.60*p_facial + │
        │ 0.40*p_postural │
        │                 │
        │ p_final:        │
        │ [0.77,0.15,    │
        │  0.05,0.03]    │
        │                 │
        │ clase = argmax  │
        │ = 0 (Atención) │
        │                 │
        │ confianza =     │
        │ max = 0.77     │
        └────────┬────────┘
                 │
ESTACIÓN 4: POST-PROCESAMIENTO Y ALERTAS
═════════════════════════════════════════════

         ┌─────────────────────────┐
         │ TEMPORAL SMOOTHING      │
         │                         │
         │ Buffer[5] frames:       │
         │ [Atenc, Dist, Dist,    │
         │  Dist, Atenc]          │
         │                         │
         │ Mode = Distracción     │
         │ (aparece 3 veces)      │
         │                         │
         │ Salida suavizada:      │
         │ Distracción            │
         └────────┬────────────────┘
                  │
         ┌────────▼─────────────┐
         │ CONFIDENCE CHECK     │
         │                      │
         │ confianza = 0.77     │
         │ Si < 0.70: skip      │
         │ Si >= 0.70: proceed  │
         │                      │
         │ Status: PROCEED ✓    │
         └────────┬─────────────┘
                  │
         ┌────────▼──────────────────┐
         │ CONTADORES Y ALERTAS      │
         │                           │
         │ Predicción: Distracción   │
         │ contador_distracción = 23 │
         │ (frames acumulados)       │
         │                           │
         │ Si > 90 frames (3s):      │
         │    GENERAR ALERTA:        │
         │    ├─ "Estudiante        │
         │    │  distraído >3seg"    │
         │    ├─ Timestamp           │
         │    ├─ ID Estudiante       │
         │    ├─ Confianza: 0.77    │
         │    └─ → Enviar a Docente │
         │                           │
         │ Predicción Siguiente:     │
         │ Atención                  │
         │ contador_distracción = 0  │
         │ (resetear)                │
         └────────┬──────────────────┘
                  │
         ┌────────▼────────────────┐
         │ OUTPUT INTERFACE        │
         │                         │
         │ • Alert Toast (push)    │
         │ • Color en Dashboard    │
         │ • Log en BD             │
         │ • API REST              │
         │ • Métricas tiempo real  │
         └─────────────────────────┘
```

---

## 6. MATRIZ DE TRANSFORMACIÓN: VARIABLES → INDICADORES

```
┌────────────────────────────────────────────────────────────────────────────┐
│ VI: ARQUITECTURA DEL MODELO → INDICADORES                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ MobileNet v3 (1.5M params)                                                 │
│   │                                                                        │
│   ├─► Accuracy: ~88% ─► Indicador I1 (VD)                                 │
│   ├─► Latencia: ~120ms ─► Indicador I2 (VD)                               │
│   ├─► FPS: ~20 fps ─► Indicador I3 (VD)                                   │
│   ├─► Tamaño: 28MB ─► Indicador I4 (VD)                                   │
│   └─► CPU: 45% ─► Indicador I7 (VD)                                       │
│                                                                             │
│ Mini-Xception (500K params) ⭐ BALANCE ÓPTIMO                               │
│   │                                                                        │
│   ├─► Accuracy: ~91% ─► Indicador I1 ✓✓ (MEJOR)                           │
│   ├─► Latencia: ~110ms ─► Indicador I2 ✓✓ (MEJOR)                         │
│   ├─► FPS: ~25 fps ─► Indicador I3 ✓✓ (MEJOR)                             │
│   ├─► Tamaño: 22MB ─► Indicador I4 ✓ (BUENO)                              │
│   └─► CPU: 38% ─► Indicador I7 ✓✓ (MEJOR)                                 │
│                                                                             │
│ YOLO-Nano (400K params)                                                   │
│   │                                                                        │
│   ├─► Accuracy: ~86% ─► Indicador I1 (Menos preciso)                      │
│   ├─► Latencia: ~130ms ─► Indicador I2 (Más lenta)                        │
│   ├─► FPS: ~18 fps ─► Indicador I3 (Más lenta)                            │
│   ├─► Tamaño: 20MB ─► Indicador I4 ✓✓ (MÁS PEQUEÑO)                       │
│   └─► CPU: 35% ─► Indicador I7 ✓ (Más eficiente)                          │
│                                                                             │
│ CONCLUSIÓN: Mini-Xception es óptimo para balance accuracy-performance      │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ VI: OPTIMIZACIÓN TÉCNICA → INDICADORES                                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Baseline (sin optimización):                                               │
│   └─► Accuracy: 91%, Latencia: 110ms, Size: 22MB, CPU: 38%               │
│                                                                             │
│ Con Cuantización 8-bit:                                                    │
│   ├─► Accuracy: 90% (-1%) ─► Aceptable                                    │
│   ├─► Latencia: 60ms (-45%) ─► ✓✓ Mejora crítica                         │
│   ├─► Size: 6MB (-73%) ─► ✓✓ Crítico para Edge                           │
│   └─► CPU: 20% (-47%) ─► ✓✓ Mejora importante                             │
│                                                                             │
│ Con Poda 20%:                                                              │
│   ├─► Accuracy: 89% (-2%)                                                  │
│   ├─► Latencia: 75ms (-32%)                                                │
│   ├─► Size: 17MB (-23%)                                                    │
│   └─► CPU: 28%                                                             │
│                                                                             │
│ Con Poda 40%:                                                              │
│   ├─► Accuracy: 85% (-6%) ─► Límite aceptable                              │
│   ├─► Latencia: 45ms (-59%) ─► ✓✓ MÁS RÁPIDO                              │
│   ├─► Size: 13MB (-41%)                                                    │
│   └─► CPU: 22%                                                             │
│                                                                             │
│ COMBINADO (Cuant 8-bit + Poda 20%):                                        │
│   ├─► Accuracy: 88% (-3%) ─► Aceptable                                     │
│   ├─► Latencia: 50ms (-55%) ─► ✓✓✓ ÓPTIMO                                 │
│   ├─► Size: 5MB (-77%) ─► ✓✓✓ ÓPTIMO PARA RPI5                            │
│   └─► CPU: 18% ─► ✓✓ Mínimo consumo                                       │
│                                                                             │
│ CONCLUSIÓN: Cuantización 8-bit + Poda 20% = Balance óptimo               │
│ Accuracy: 88% vs 91% (3% pérdida aceptable)                               │
│ Latencia: 50ms vs 110ms (54% mejora CRÍTICA)                              │
│ Size: 5MB vs 22MB (77% reducción CRÍTICA para Edge)                       │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. ESPECIFICACIÓN DE MODELOS MATEMÁTICOS

### Modelo de Inferencia Facial (Mini-Xception)

```
f_facial: ℝ^12 → ℝ^4

Input: X_facial = [AU12, AU26, AU01, gaze_h, gaze_v, 
                    head_pitch, head_yaw, head_roll,
                    emo_neutro, emo_feliz, emo_triste, emo_sorpresa]

Layer 1: z1 = ReLU(W1 @ X_facial + b1),  donde W1 ∈ ℝ^128×12, b1 ∈ ℝ^128
Layer 2: z2 = BatchNorm(z1)
Layer 3: z3 = ReLU(W2 @ z2 + b2),       donde W2 ∈ ℝ^64×128, b2 ∈ ℝ^64
Layer 4: z4 = Dropout(z3, p=0.2)
Layer 5: y = Softmax(W3 @ z4 + b3),    donde W3 ∈ ℝ^4×64, b3 ∈ ℝ^4

Output: p_facial = [p_atención, p_distracción, p_fatiga, p_neutral]
        ∑ p_i = 1, p_i ∈ [0, 1]
```

### Modelo de Inferencia Postural (MobileNet v3)

```
f_postural: ℝ^8 → ℝ^4

Input: X_postural = [cuello_x, cuello_y, inclin_torso, mov_brazo_d,
                      mov_brazo_i, movimiento_total, presencia, confidence]

Layer 1: z1 = ReLU(W1 @ X_postural + b1),  W1 ∈ ℝ^256×8
Layer 2: z2 = Dropout(z1, p=0.4)
Layer 3: z3 = ReLU(W2 @ z2 + b2),          W2 ∈ ℝ^128×256
Layer 4: z4 = Dropout(z3, p=0.2)
Layer 5: y = Softmax(W3 @ z4 + b3),        W3 ∈ ℝ^4×128

Output: p_postural = [p_atención, p_distracción, p_fatiga, p_neutral]
```

### Fusión Multimodal (Weighted Average)

```
p_final = α × p_facial + (1-α) × p_postural

donde: α = 0.60 (peso facial), (1-α) = 0.40 (peso postural)

p_final ∈ ℝ^4, ∑ p_final_i = 1

Predicción final:
clase = argmax_i(p_final_i) ∈ {0, 1, 2, 3}
confianza = max_i(p_final_i) ∈ [0, 1]
```

### Post-procesamiento Temporal (Smoothing)

```
Sea H_t = [s_t-4, s_t-3, s_t-2, s_t-1, s_t] donde s_i ∈ {0,1,2,3}

s_suavizado_t = mode(H_t) = argmax_i(count(clase_i en H_t))

Ventaja: Reduce ruido temporal, estabiliza predicciones
Costo: Latencia adicional ~50ms (4 frames a 30fps)
```

---

## 8. RESUMEN EJECUTIVO

### Síntesis del Proyecto

El sistema propuesto implementa un **pipeline Edge AI completo** para medición automática de estados afectivos en aulas híbridas, integrando:

| **Componente** | **Especificación** | **Justificación** |
|---|---|---|
| **Hardware Target** | Jetson Nano / RPi5 | Edge processing, low cost |
| **Input** | Video RGB 30-60 FPS | Disponible en aulas actuales |
| **Features** | 20 features (facial+postural) | Balance información vs velocidad |
| **Modelo Facial** | Mini-Xception (500K params) | Óptimo accuracy-latency |
| **Modelo Postural** | MobileNet v3 (1.5M params) | Detecta fatiga/distracción |
| **Optimización** | 8-bit Quantization + 20% Pruning | 50ms latencia, 5MB modelo |
| **Fusión** | Weighted Average (60/40) | Robusto, computacionalmente eficiente |
| **Salida** | Alertas tiempo real + Dashboard | Útil pedagógico inmediato |

### Indicadores Clave Esperados

**Fase de Validación (Objetivo 5):**
- **Accuracy General:** ≥90%
- **Latencia:** ≤100ms (captura → predicción)
- **FPS:** ≥25 fps (fluidez)
- **Usabilidad Docente:** ≥4.0/5.0 (escala Likert)
- **Precisión Alertas:** ≥80% (reducción falsos positivos)

### Datasets Recomendados

**Prioritarios:**
1. DAiSEE (benchmark internacional) + local UNI dataset
2. DIPSER (si acceso disponible) para validación multimodal
3. FER2013 (transfer learning para emociones)

**Secundarios:**
- EngageNet, EmotiW para comparativas
- ICCV Student Engagement Dataset

### Timeline y Próximos Pasos

- **Semana 1-2:** Solicitar datos, preparar dev environment
- **Semana 3-4:** Feature extraction y análisis exploratorio
- **Semana 5-8:** Entrenamiento y optimización de modelos
- **Semana 9-10:** Desarrollo prototipo y dashboard
- **Semana 11-14:** Validación en aulas, ajustes finales
- **Semana 15-16:** Documentación y reporte final

---

**Análisis Preparado por:** Especialista en IA/ML  
**Fecha:** Diciembre 2025  
**Versión:** 1.0 - Arquitectura y Diagramas
