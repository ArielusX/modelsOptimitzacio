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

nodes_list = [1,2,4,8,16,32]
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

# Graficar
if results:
    nodes, times = zip(*results)
    plt.plot(nodes, times, marker='o')
    plt.xlabel("Número de nodos")
    plt.ylabel("Tiempo total (makespan) [s]")
    plt.xscale('log', base=2)
    plt.title("Benchmark entrenamiento distribuido")
    plt.grid(True)
    plt.show()
else:
    print("No hay resultados válidos para graficar")
