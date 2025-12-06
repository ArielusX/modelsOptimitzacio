import pandas as pd
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import shutil

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

# Separar X e y
X = df.drop(target_col, axis=1)
y = df[target_col]

if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError("Dataset final vacío o inválido")

nodes_list = [2,4,8,16,32]
results = []

# Carpetas temporales
temp_dir = "chunks"
results_dir = "results"
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

for n_nodes in nodes_list:
    print(f"\n--- Probando {n_nodes} nodos ---")
    splits = np.array_split(np.arange(len(X)), n_nodes)

    # Guardar chunks CSV
    chunk_files = []
    for i, idx in enumerate(splits):
        chunk_df = df.iloc[idx]
        if chunk_df.shape[0] == 0:
            print(f"Chunk {i} vacío, se salta")
            continue
        chunk_file = os.path.join(temp_dir, f"dataset_chunk_{i}.csv")
        chunk_df.to_csv(chunk_file, index=False)
        chunk_files.append(chunk_file)

    # Lanzar contenedores
    procs = []
    for chunk in chunk_files:
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{os.path.abspath(temp_dir)}:/data",       # CSV dentro del contenedor
            "-v", f"{os.path.abspath(results_dir)}:/results", # resultados dentro del contenedor
            "ml_node",
            "python", "/app/train_node.py", f"/data/{os.path.basename(chunk)}"
        ]
        p = subprocess.Popen(cmd)
        procs.append(p)

    # Esperar a que todos terminen
    for p in procs:
        p.wait()

    # Leer resultados
    times = []
    for chunk in chunk_files:
        result_file = os.path.join(results_dir, f"result_{os.path.splitext(os.path.basename(chunk))[0]}.txt")
        if not os.path.exists(result_file):
            print(f"Warning: {result_file} no existe, se salta")
            continue
        with open(result_file) as f:
            times.append(float(f.read().strip()))
        os.remove(result_file)  # limpiar

    if len(times) == 0:
        print(f"No hubo resultados válidos para {n_nodes} nodos, se salta")
        continue

    makespan = max(times)
    print(f"Makespan para {n_nodes} nodos: {makespan:.2f} s")
    results.append((n_nodes, makespan))

# Limpiar carpeta temporal
shutil.rmtree(temp_dir)
import seaborn as sns
import psutil
import time

import numpy as np
import matplotlib.pyplot as plt

# ============================
#   CÁLCULO VARIABLES MODELO
# ============================

if results:
    nodes, times = zip(*results)
    nodes = np.array(nodes, dtype=float)
    times = np.array(times, dtype=float)

    # Tamaño del dataset (número de muestras)
    D = len(df)          # filas del dataframe
    E = 1                # RF no tiene epochs -> 1 pasada lógica

    # Throughput global y por nodo (reales)
    throughput_global = D * E / times            # muestras/s
    throughput_per_node = throughput_global / nodes  # muestras/s/nodo

    # Usamos el mejor throughput por nodo como referencia teórica de cómputo
    ref_throughput_node = throughput_per_node.max()

    # Tiempo ideal de cómputo si SOLO hubiera cálculo y nada de comunicación
    ideal_compute_time = D * E / (ref_throughput_node * nodes)

    # Coste de comunicación (en segundos) según el modelo
    comm_cost = np.maximum(times - ideal_compute_time, 0.0)

    # "Latencia efectiva" por nodo (L * C_sync) en segundos y ms
    latency_effective = comm_cost / nodes          # s
    latency_effective_ms = latency_effective * 1000

    # Métricas clásicas de HPC
    # (usamos el primer T(N) como referencia de speedup)
    T_ref = times[0]
    speedup = T_ref / times
    efficiency = speedup / nodes
    overhead = nodes * times - T_ref
    time_per_node = times / nodes

    # ============================
    #   IMPRIMIR TABLA DE VARIABLES
    # ============================

    print("\n===== VARIABLES DEL MODELO (DEFINICIÓN) =====\n")
    print("T : Temps total d’entrenament              (segons)")
    print("A : Eficàcia del model                     (%)")
    print(f"N : Nombre de màquines o nodes            {list(map(int, nodes))}")
    print("L : Latència efectiva de comunicació       (ms, derivada de T(N))")
    print("B : Batch size global                      (no aplicable en RF)")
    print(f"D : Mida del conjunt de dades             {D} mostres")
    print(f"E : Nombre d’epochs                       {E} (RF ≈ 1 passada)")
    print(f"\nThroughput global real (mostres/s) per N : {throughput_global}")
    print(f"Throughput per node real (mostres/s/node): {throughput_per_node}")
    print(f"Throughput per node de referència         : {ref_throughput_node:.2f} mostres/s/node\n")

    print("===== RESULTADOS CALCULADOS PER CONFIGURACIÓN =====\n")
    for n, t, s, e, o, tp, tc, l_ms in zip(
        nodes, times, speedup, efficiency, overhead, time_per_node, comm_cost, latency_effective_ms
    ):
        print(f"Nodos = {int(n)}")
        print(f"  T(N)             = {t:.4f} s")
        print(f"  Speedup          = {s:.4f}")
        print(f"  Eficiencia       = {e:.4f}")
        print(f"  Overhead         = {o:.4f}")
        print(f"  Tiempo/Nodo      = {tp:.4f} s")
        print(f"  Coste comunicación T_comm(N) = {tc:.4f} s")
        print(f"  Latencia efectiva por nodo   = {l_ms:.4f} ms\n")

    # ============================
    #   GRAFICOS (TUS 5 GRÁFICOS)
    # ============================

    # Punto dulce (mínimo T(N))
    min_time = times.min()
    sweet_idx = times.argmin()
    sweet_nodes = int(nodes[sweet_idx])

    # -----------------------------
    #     GRAFICO 1: MAKESPAN
    # -----------------------------
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
    plt.title("Makespan del Entrenamiento Distribuido", fontsize=18)
    plt.xlabel("Número de nodos (log2)", fontsize=14)
    plt.ylabel("Tiempo total (s)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    #     GRAFICO 2: SPEEDUP
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, speedup, marker='o', linewidth=2.5, color="green")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("Speedup S(N) = T(ref) / T(N)", fontsize=18)
    plt.xlabel("Número de nodos (log2)", fontsize=14)
    plt.ylabel("Speedup", fontsize=14)
    plt.xscale("log", base=2)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    #     GRAFICO 3: EFICIENCIA
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, efficiency, marker='o', linewidth=2.5, color="orange")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("Eficiencia E(N) = S(N) / N", fontsize=18)
    plt.xlabel("Número de nodos (log2)", fontsize=14)
    plt.ylabel("Eficiencia", fontsize=14)
    plt.xscale("log", base=2)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    #     GRAFICO 4: OVERHEAD
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, overhead, marker='o', linewidth=2.5, color="purple")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("Overhead O(N) = N·T(N) – T(ref)", fontsize=18)
    plt.xlabel("Número de nodos (log2)", fontsize=14)
    plt.ylabel("Overhead", fontsize=14)
    plt.xscale("log", base=2)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    #     GRAFICO 5: TIEMPO POR NODO
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, time_per_node, marker='o', linewidth=2.5, color="blue")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("Tiempo Promedio por Nodo", fontsize=18)
    plt.xlabel("Número de nodos (log2)", fontsize=14)
    plt.ylabel("T(N) / N (s)", fontsize=14)
    plt.xscale("log", base=2)
    plt.tight_layout()
    plt.show()

    # (Opcional) GRAFICO 6: COSTE DE COMUNICACIÓN
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, comm_cost, marker='o', linewidth=2.5, color="red")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("Coste de Comunicación T_comm(N)", fontsize=18)
    plt.xlabel("Número de nodos (log2)", fontsize=14)
    plt.ylabel("Tiempo de comunicación (s)", fontsize=14)
    plt.xscale("log", base=2)
    plt.tight_layout()
    plt.show()

else:
    print("No hay resultados válidos para graficar")
