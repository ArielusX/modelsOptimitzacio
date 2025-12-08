import pandas as pd
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Dataset
dataset = "curse.csv"
df = pd.read_csv(dataset)

# Procesar target
target_col = 'Completed'
df[target_col] = df[target_col].astype(str).str.strip().map({
    'Completed': 1,
    'Not Completed': 0
})
df = df.dropna(subset=[target_col])

# Columnas categóricas → one-hot
categorical_cols = ['Gender', 'Education_Level', 'Employment_Status', 
                    'City', 'Device_Type', 'Internet_Connection_Quality', 
                    'Course_Level', 'Payment_Mode']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Eliminar IDs irrelevantes
irrelevant_cols = ['Student_ID', 'Name', 'Course_ID', 'Course_Name', 'Category',
                   'Enrollment_Date', 'Fee_Paid', 'Discount_Used']
df = df.drop(columns=[c for c in irrelevant_cols if c in df.columns], errors='ignore')

# Reemplazar NaN en features por 0
df = df.fillna(0)

# Separar X e y (dataset completo para evaluación)
X_full = df.drop(target_col, axis=1)
y_full = df[target_col]

if X_full.shape[0] == 0 or y_full.shape[0] == 0:
    raise ValueError("Dataset final vacío o inválido")

print(f"Dataset cargado: {len(df)} muestras, {X_full.shape[1]} features")

nodes_list = [1, 2, 4, 8, 16, 32]
results = []
mean_accs = []  # Lista para guardar los accuracies

# Carpetas temporales
temp_dir = "chunks"
results_dir = "results"
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

for n_nodes in nodes_list:
    print(f"\n{'='*60}")
    print(f"Probando con {n_nodes} nodos")
    print(f"{'='*60}")
    
    splits = np.array_split(np.arange(len(X_full)), n_nodes)

    # Guardar chunks CSV
    chunk_files = []
    for i, idx in enumerate(splits):
        chunk_df = df.iloc[idx]
        if chunk_df.shape[0] == 0:
            print(f"⚠️  Chunk {i} vacío, se salta")
            continue
        chunk_file = os.path.join(temp_dir, f"dataset_chunk_{i}.csv")
        chunk_df.to_csv(chunk_file, index=False)
        chunk_files.append(chunk_file)
        print(f"✓ Chunk {i}: {len(chunk_df)} muestras guardadas")

    # Lanzar contenedores (montar también el dataset completo)
    procs = []
    print(f"\n🚀 Lanzando {len(chunk_files)} contenedores Docker...")
    for idx, chunk in enumerate(chunk_files):
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{os.path.abspath(temp_dir)}:/data",
            "-v", f"{os.path.abspath(results_dir)}:/results",
            "-v", f"{os.path.abspath(dataset)}:/data/../curse.csv:ro",  # Montar dataset completo
            "ml_node",
            "python", "/app/train_node.py", f"/data/{os.path.basename(chunk)}"
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append(p)

    # Esperar a que todos terminen
    print(f"⏳ Esperando a que completen los {len(procs)} nodos...")
    for i, p in enumerate(procs):
        p.wait()
        print(f"✓ Nodo {i} completado")

    # Leer resultados (tiempos y predicciones)
    times = []
    pred_files = []
    for chunk in chunk_files:
        chunk_basename = os.path.splitext(os.path.basename(chunk))[0]
        result_file = os.path.join(results_dir, f"result_{chunk_basename}.txt")
        pred_file = os.path.join(results_dir, f"predictions_{chunk_basename}.npy")
        
        if not os.path.exists(result_file):
            print(f"⚠️  Warning: {result_file} no existe, se salta")
            continue
        
        with open(result_file) as f:
            times.append(float(f.read().strip()))
        
        if os.path.exists(pred_file):
            pred_files.append(pred_file)
        
        os.remove(result_file)  # limpiar archivo de resultado

    if len(times) == 0:
        print(f"❌ No hubo resultados válidos para {n_nodes} nodos, se salta")
        continue

    makespan = max(times)
    print(f"\n📊 Makespan para {n_nodes} nodos: {makespan:.2f}s")
    print(f"   Tiempo promedio por nodo: {np.mean(times):.2f}s")
    
    # Agregar predicciones y calcular accuracy en el dataset completo
    print(f"\n🔄 Agregando predicciones de {len(pred_files)} modelos...")
    if len(pred_files) > 0:
        # Cargar todas las predicciones
        all_predictions = [np.load(pf) for pf in pred_files]
        
        # Votación por mayoría (ensemble)
        predictions_array = np.array(all_predictions)
        y_pred_ensemble = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), 
            axis=0, 
            arr=predictions_array
        )
        
        # Calcular accuracy sobre el dataset completo
        accuracy = accuracy_score(y_full, y_pred_ensemble)
        mean_accs.append(accuracy)
        print(f"✓ Accuracy del modelo agregado: {accuracy:.4f}")
        
        # Limpiar archivos de predicciones
        for pf in pred_files:
            os.remove(pf)
    else:
        print(f"⚠️  No se encontraron predicciones, accuracy = 0")
        mean_accs.append(0.0)

    results.append((n_nodes, makespan))

# Limpiar carpeta temporal
shutil.rmtree(temp_dir)

# ============================
#   CÁLCULO VARIABLES MODELO
# ============================

if results:
    print(f"\n{'='*60}")
    print("ANÁLISIS DE RESULTADOS")
    print(f"{'='*60}\n")
    
    nodes = np.array([r[0] for r in results], dtype=float)
    times = np.array([r[1] for r in results], dtype=float)
    mean_accs = np.array(mean_accs)

    # Tamaño del dataset (número de muestras)
    D = len(df)
    E = 1

    # Throughput global y por nodo (reales)
    throughput_global = D * E / times
    throughput_per_node = throughput_global / nodes

    # Usamos el mejor throughput por nodo como referencia teórica
    ref_throughput_node = throughput_per_node.max()

    # Tiempo ideal de cómputo
    ideal_compute_time = D * E / (ref_throughput_node * nodes)

    # Coste de comunicación
    comm_cost = np.maximum(times - ideal_compute_time, 0.0)

    # Latencia efectiva
    latency_effective = comm_cost / nodes
    latency_effective_ms = latency_effective * 1000

    # Métricas HPC
    T_ref = times[0]
    speedup = T_ref / times
    efficiency = speedup / nodes
    overhead = nodes * times - T_ref
    time_per_node = times / nodes
    
    # Encontrar sweet spot
    sweet_idx = np.argmin(times)
    sweet_nodes = int(nodes[sweet_idx])
    min_time = times[sweet_idx]

    # Tabla resumen
    print(f"{'Nodos':>8} {'Tiempo(s)':>12} {'Speedup':>10} {'Eficiencia':>12} {'Accuracy':>10}")
    print("-" * 60)
    for i in range(len(nodes)):
        print(f"{int(nodes[i]):>8} {times[i]:>12.2f} {speedup[i]:>10.2f} {efficiency[i]:>12.2%} {mean_accs[i]:>10.4f}")
    print(f"\n🎯 Sweet spot: {sweet_nodes} nodos con {min_time:.2f}s")

    # ============================
    #   GRÁFICOS
    # ============================

    # 1. MAKESPAN
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, times, marker='o', linewidth=2.5)
    plt.scatter([sweet_nodes], [min_time], s=180, color='crimson', zorder=5)
    plt.annotate(
        f"Sweet spot: {sweet_nodes} nodos\n{min_time:.2f} s",
        xy=(sweet_nodes, min_time),
        xytext=(sweet_nodes * 1.1, min_time * 1.1),
        fontsize=12,
        color='crimson',
        arrowprops=dict(arrowstyle="->", color='crimson', lw=1.5)
    )
    plt.xscale("log", base=2)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.title("1. Makespan del Entrenamiento Distribuido", fontsize=18)
    plt.xlabel("Número de nodos (log2)", fontsize=14)
    plt.ylabel("Tiempo total (s)", fontsize=14)
    plt.tight_layout()
    plt.savefig("1_makespan.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 2. SPEEDUP
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, speedup, marker='o', linewidth=2.5, color="green")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("2. Speedup S(N) = T(ref) / T(N)", fontsize=18)
    plt.xlabel("Número de nodos (log2)", fontsize=14)
    plt.ylabel("Speedup", fontsize=14)
    plt.xscale("log", base=2)
    plt.tight_layout()
    plt.savefig("2_speedup.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 3. EFICIENCIA
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, efficiency, marker='o', linewidth=2.5, color="orange")
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label="Eficiencia ideal")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("3. Eficiencia E(N) = S(N) / N", fontsize=18)
    plt.xlabel("Número de nodos (log2)", fontsize=14)
    plt.ylabel("Eficiencia", fontsize=14)
    plt.xscale("log", base=2)
    plt.tight_layout()
    plt.savefig("3_eficiencia.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 4. OVERHEAD
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, overhead, marker='o', linewidth=2.5, color="purple")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("4. Overhead O(N) = N·T(N) – T(ref)", fontsize=18)
    plt.xlabel("Número de nodos (log2)", fontsize=14)
    plt.ylabel("Overhead (s)", fontsize=14)
    plt.xscale("log", base=2)
    plt.tight_layout()
    plt.savefig("4_overhead.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 5. TIEMPO POR NODO
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, time_per_node, marker='o', linewidth=2.5, color="blue")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("5. Tiempo Promedio por Nodo", fontsize=18)
    plt.xlabel("Número de nodos (log2)", fontsize=14)
    plt.ylabel("T(N) / N (s)", fontsize=14)
    plt.xscale("log", base=2)
    plt.tight_layout()
    plt.savefig("5_tiempo_por_nodo.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 6. COSTE DE COMUNICACIÓN
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, comm_cost, marker='o', linewidth=2.5, color="red")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("6. Coste de Comunicación T_comm(N)", fontsize=18)
    plt.xlabel("Número de nodos (log2)", fontsize=14)
    plt.ylabel("Tiempo de comunicación (s)", fontsize=14)
    plt.xscale("log", base=2)
    plt.tight_layout()
    plt.savefig("6_coste_comunicacion.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 7. LATENCIA EFECTIVA
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, latency_effective_ms, marker='o', linewidth=2.5, color="darkcyan")
    plt.xscale("log", base=2)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("7. Latencia Efectiva por Nodo (T_comm(N)/N)", fontsize=18)
    plt.xlabel("Número de nodos (log2)", fontsize=14)
    plt.ylabel("Latencia efectiva (ms)", fontsize=14)
    plt.tight_layout()
    plt.savefig("7_latencia_efectiva.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 8. ACCURACY
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, mean_accs, marker='o', linewidth=2.5, color="black")
    plt.xscale("log", base=2)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("8. Accuracy del Modelo Agregado", fontsize=18)
    plt.xlabel("Número de nodos (log2)", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("8_accuracy.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✅ Gráficos guardados exitosamente")

else:
    print("\n❌ No hay resultados válidos para graficar")