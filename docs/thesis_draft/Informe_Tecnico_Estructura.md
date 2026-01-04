# Estructura del Informe Técnico (Guía No. 2)
**Proyecto:** Medición Automática de Estados Afectivos en Aulas Híbridas (Edge AI)

Este documento describe la estructura y contenido requerido para el Informe Técnico Final, basado en la "Guía No. 2 Desarrollo".

---

## CAPÍTULO I. PLANTEAMIENTO DEL ESTUDIO

### 1.1 Planteamiento del Problema
- **Problema:** Dificultad para monitorear atención en aulas híbridas.
- **Contexto:** Limitaciones de latencia y privacidad en soluciones cloud.
- **Justificación:** Necesidad de Feedback en tiempo real.

### 1.2 Formulación del Problema
- ¿Cómo medir automáticamente estados afectivos con baja latencia y alta privacidad?

### 1.3 Objetivos
- **Objetivo General:** (Ver README)
- **Objetivos Específicos (5):**
  1. Adquisición de Datos
  2. Extracción de Features
  3. Optimización
  4. Prototipo
  5. Validación (Técnica y Pedagógica)

### 1.4 Variables e Indicadores
- **Independientes:** Arquitectura (MobileNet vs Xception), Optimización (Int8 vs Float32).
- **Dependientes:** Accuracy, Latencia, FPS, Usabilidad.
- **Indicadores:** Kappa de Cohen, F1-Score, ms/frame.

---

## CAPÍTULO II. MARCO TEÓRICO

### 2.1 Antecedentes
- Papers sobre Affective Computing (DAiSEE, FER2013).
- Soluciones Edge AI previas.

### 2.2 Bases Teóricas
- Redes Neuronales Convolucionales (CNNs).
- Edge Computing & Quantization (TFLite).
- Fusión Multimodal (Late Fusion).

---

## CAPÍTULO III. METODOLOGÍA (MATERIAL Y MÉTODOS)

### 3.1 Diseño de Investigación
- Experimental, cuantitativo.

### 3.2 Población y Muestra (Dataset)
- Descripción de FER2013 / Dataset Propio.
- Criterios de exclusión/inclusión.

### 3.3 Técnicas e Instrumentos (Pipelilne)
1. **Adquisición:** `src/data`
2. **Procesamiento:** `src/features`
3. **Análisis:** `src/models`

### 3.4 Procedimiento de Optimización
- Descripción del proceso de poda y cuantización `src/models/optimize.py`.

---

## CAPÍTULO IV. RESULTADOS Y DISCUSIÓN

### 4.1 Resultados de Entrenamiento (OE3)
- Curvas de aprendizaje (Loss/Accuracy).
- Comparativa MobileNet vs Mini-Xception.

### 4.2 Resultados de Validación (OE5)
- Matriz de Confusión.
- Tabla de Latencias (Edge Device).

### 4.3 Discusión
- Análisis de trade-off Precisión vs Velocidad.
- Comparación con estado del arte.

---

## CAPÍTULO V. CONCLUSIONES Y RECOMENDACIONES

### 5.1 Conclusiones
- Se logró X% de precisión con Y ms de latencia.
- La cuantización redujo el tamaño en Z%.

### 5.2 Recomendaciones
- Ampliar dataset local.
- Explorar Vision Transformers ligeros.

---

## REFERENCIAS
- Formato APA 7ma Edición.

## ANEXOS
- Manual de Usuario (Setup, Dashboard).
- Snippets de Código Clave.
