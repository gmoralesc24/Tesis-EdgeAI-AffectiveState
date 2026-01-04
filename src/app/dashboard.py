import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
# from src.app.inference import InferenceEngine # Comentado para evitar error de import si no corre desde root

def main():
    st.set_page_config(page_title="Monitor de Atención Edge AI", layout="wide")
    
    st.title("🎓 Monitor de Atención en Aulas Híbridas (Edge AI)")
    st.markdown("Sistema de medición automática de estados afectivos mediante análisis multimodal.")
    
    # Sidebar configuración
    st.sidebar.header("Configuración")
    model_source = st.sidebar.selectbox("Modelo", ["MobileNetV3", "Mini-Xception (Int8)"])
    camera_idx = st.sidebar.number_input("Índice de Cámara", 0, 5, 0)
    confidence_threshold = st.sidebar.slider("Umbral de Confianza", 0.0, 1.0, 0.7)
    
    # Layout Principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Video en Tiempo Real")
        cam_placeholder = st.empty()
        
    with col2:
        st.subheader("Métricas en Vivo")
        metric_status = st.empty()
        metric_conf = st.empty()
        metric_fps = st.empty()
        
        st.subheader("Historial")
        chart_placeholder = st.line_chart([])
        
    start_btn = st.sidebar.button("Iniciar Monitor")
    stop_btn = st.sidebar.button("Detener")
    
    if start_btn:
        cap = cv2.VideoCapture(camera_idx)
        history_data = []
        
        # engine = InferenceEngine("path_to_model") # Mock por ahora
        
        while not stop_btn:
            ret, frame = cap.read()
            if not ret:
                st.error("Error al leer cámara")
                break
                
            # Simulación de inferencia
            start_t = time.time()
            # result = engine.predict(frame)
            # Fake result para demo UI
            states = ["Atencion", "Distraccion", "Fatiga"]
            current_state = np.random.choice(states, p=[0.7, 0.2, 0.1])
            conf = np.random.uniform(0.6, 0.99)
            process_time = (time.time() - start_t) * 1000
            
            # Dibujar en frame
            color = (0, 255, 0) if current_state == "Atencion" else (0, 0, 255)
            cv2.putText(frame, f"{current_state} ({conf:.2f})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Actualizar UI
            cam_placeholder.image(frame, channels="BGR")
            
            metric_status.metric("Estado Actual", current_state, delta_color="normal")
            metric_conf.metric("Confianza", f"{conf:.2f}")
            metric_fps.metric("Latencia", f"{process_time:.1f} ms")
            
            history_data.append(1 if current_state == "Atencion" else 0)
            if len(history_data) > 50: history_data.pop(0)
            chart_placeholder.line_chart(history_data)
            
            time.sleep(0.03)
            
        cap.release()

if __name__ == "__main__":
    main()
