# Comparativa Técnica: Transfer Learning vs Entrenamiento desde Cero

Este documento sirve de base para el **Capítulo III (Metodología)** o **Capítulo IV (Discusión)** de la tesis.

## 1. Definiciones Conceptuales

### A. Entrenamiento desde Cero (Training from Scratch)
**Concepto:** Se inicializa la red neuronal con pesos aleatorios. La red debe aprender absolutamente todo desde el principio: desde cómo detectar una línea o un borde (capas iniciales) hasta cómo identificar una "cara de enojo" (capas finales).
*   **Ejemplo en el proyecto:** Modelo **Mini-Xception**.
*   **Analogía:** Es como enseñar a un bebé a leer. Primero debe aprender las formas de las letras, luego sílabas, luego palabras y finalmente el significado.

### B. Transferencia de Aprendizaje (Transfer Learning)
**Concepto:** Se utiliza una red neuronal que ya ha sido entrenada previamente en un dataset masivo (como **ImageNet**, con 14 millones de imágenes). Esta red ya "sabe ver" (detectar bordes, texturas, formas geométricas). Solo re-entrenamos las últimas capas para adaptarlas a nuestra tarea específica (emociones).
*   **Ejemplo en el proyecto:** Modelo **MobileNetV3 Small**.
*   **Analogía:** Es como pedirle a un adulto que aprenda a diferenciar señales de tráfico de otro país. Ya sabe ver y leer forms, solo necesita aprender el significado específico de las nuevas señales.

---

## 2. Comparativa para el Proyecto (Tesis Edge AI)

| Característica | Mini-Xception (Scratch) | MobileNetV3 (Transfer Learning) |
| :--- | :--- | :--- |
| **Tiempo de Entrenamiento** | **Alto (+50 épocas)**. Necesita muchas repeticiones para aprender características básicas. | **Bajo (10-20 épocas)**. Converge rápido porque ya tiene conocimiento base. |
| **Dependencia de Datos** | **Alta**. Requiere miles de imágenes para generalizar bien y no memorizar (overfitting). | **Media/Baja**. Funciona sorprendentemente bien incluso con datasets pequeños. |
| **Recursos Computacionales** | Requiere más cómputo durante más tiempo. | Requiere menos tiempo de GPU. |
| **Precisión Inicial** | Empieza adivinando al azar (~25%). | Empieza con una base sólida, la curva de aprendizaje sube rápido. |
| **Idoneidad para Tesis** | Bueno para demostrar conocimiento de arquitecturas CNN. | **Excelente** para demostrar eficiencia y resultados robustos en Edge AI. |

## 3. Justificación de la Selección

Para este proyecto de tesis enfocado en **Edge AI**, se prioriza **MobileNetV3 con Transfer Learning** por dos razones críticas:

1.  **Eficiencia Energética y Temporal:** Al converger más rápido, reducimos el consumo de energía en el entrenamiento y permitimos iteraciones más rápidas de desarrollo.
2.  **Robustez:** Al estar pre-entrenado en millones de imágenes variadas, MobileNetV3 suele ser más robusto ante variaciones de iluminación o ruido en el aula que una red entrenada solo con un dataset limitado como FER2013.

Sin embargo, se mantiene **Mini-Xception** como segunda opción para validar si una arquitectura especializada y más pequeña puede competir en latencia (ms) contra un modelo generalista optimizado.
