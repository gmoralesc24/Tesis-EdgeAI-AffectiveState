# Sistema Edge AI para Medición de Estados Afectivos en Aulas Híbridas

## Descripción del Proyecto
Este proyecto es parte de una tesis de maestría (MIA-UNI) que busca desarrollar una solución **Edge AI Multimodal** para monitorear automáticamente la atención y estados afectivos (Atención, Distracción, Fatiga) de estudiantes en entornos de educación híbrida.

La solución se centra en la privacidad, baja latencia y eficiencia computacional, utilizando análisis facial y postural en dispositivos como Jetson Nano o Raspberry Pi.

## Objetivos Específicos

1.  **OE1: Adquisición y Normalización de Datos:** Creación de dataset multimodal en aulas reales.
2.  **OE2: Extracción de Características:** Análisis facial (landmarks, mirada) y postural.
3.  **OE3: Optimización de Modelos:** Modelos ligeros (MobileNet, YOLO-Nano) cuantizados para Edge.
4.  **OE4: Prototipo y Dashboard:** Sistema de alertas en tiempo real para docentes.
5.  **OE5: Validación:** Pruebas de campo y análisis de impacto pedagógico.

## Estructura del Proyecto

```
Proy_Repo/
├── docs/                   # Documentación (Diagramas, Reportes, Guías)
├── src/                    # Código Fuente
│   ├── data/               # Scripts de captura y preprocesamiento
│   ├── features/           # Extracción facial y postural
│   ├── models/             # Entrenamiento y optimización (TFLite/ONNX)
│   ├── app/                # Aplicación final y Dashboard
│   └── evaluation/         # Scripts de validación y métricas
├── tests/                  # Pruebas unitarias
└── README.md               # Este archivo
```

## Requisitos
Ver `requirements.txt`.

## Autor
Tesista - Maestría en Inteligencia Artificial, UNI.
