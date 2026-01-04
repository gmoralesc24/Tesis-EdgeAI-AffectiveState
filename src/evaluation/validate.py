import argparse
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def validate_system(test_data_path=None, model_path=None):
    """
    Ejecuta validación del sistema completo y genera reporte.
    
    Args:
        test_data_path (str): Ruta al dataset de prueba (si aplica).
        model_path (str): Ruta al modelo entrenado (o TFLite).
    """
    print("Iniciando Validación del Sistema Edge AI...")
    
    # Simulación de datos de prueba si no se proveen (para demostrar funcionalidad)
    if test_data_path is None:
        print("NOTA: Usando datos sintéticos para demostración de métricas.")
        y_true = np.random.randint(0, 4, size=100)
        # Simular predicciones con 85% de accuracy
        y_pred = y_true.copy()
        noise_idx = np.random.choice(100, 15, replace=False)
        y_pred[noise_idx] = np.random.randint(0, 4, size=15)
        
        # Simular latencias
        latencies = np.random.normal(loc=85, scale=10, size=100) # media 85ms
    else:
        # Aquí cargaríamos el dataset real y correríamos inference_engine.predict()
        pass

    # 1. Métricas de Clasificación
    target_names = ['Atencion', 'Distraccion', 'Fatiga', 'Neutral']
    report = classification_report(y_true, y_pred, target_names=target_names)
    matrix = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    # 2. Métricas de Performance (Edge)
    avg_latency = np.mean(latencies)
    fps = 1000.0 / avg_latency
    p95_latency = np.percentile(latencies, 95)
    
    # 3. Mostrar Reporte
    print("\n" + "="*40)
    print(" RESULTADOS DE VALIDACIÓN (OE5)")
    print("="*40)
    print(f"\nExactitud Global (Accuracy): {acc:.2%}")
    print("\nReporte de Clasificación:\n")
    print(report)
    print("\nMatriz de Confusión:\n")
    print(matrix)
    print("\n" + "-"*40)
    print(" PERFORMANCE DE EJECUCIÓN")
    print("-"*40)
    print(f"Latencia Promedio: {avg_latency:.2f} ms")
    print(f"Latencia P95:      {p95_latency:.2f} ms")
    print(f"FPS Estimado:      {fps:.2f} fps")
    print("="*40)
    
    # Validación contra objetivos
    ok_acc = "CUMPLE" if acc >= 0.90 else "NO CUMPLE (>90%)"
    ok_lat = "CUMPLE" if avg_latency <= 100 else "NO CUMPLE (<100ms)"
    
    print(f"\nEvaluación vs Objetivos:")
    print(f"1. Precisión: {ok_acc}")
    print(f"2. Latencia:  {ok_lat}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Ruta al modelo")
    parser.add_argument("--test_data", type=str, help="Ruta a datos de test")
    
    args = parser.parse_args()
    validate_system(args.test_data, args.model)
