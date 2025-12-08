import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time
import sys
import os

# Argumento: CSV chunk
csv_file = sys.argv[1]

print(f"[NODE] Iniciando con chunk: {csv_file}")

# Leer CSV
df = pd.read_csv(csv_file)
print(f"[NODE] Chunk cargado: {len(df)} muestras")

# Target
target_col = 'Completed'
if df[target_col].dtype == 'object':
    df[target_col] = df[target_col].map({'Completed': 1, 'Not Completed': 0})

# Seleccionar solo features numéricas
X = df.drop(target_col, axis=1).select_dtypes(include=[np.number])
y = df[target_col]

print(f"[NODE] Features: {X.shape[1]}, Muestras: {len(X)}")

# Configuración Random Forest
rf_params = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_leaf': 1,
    'n_jobs': 1
}

# Entrenamiento
start = time.time()
model = RandomForestClassifier(**rf_params)
model.fit(X, y)
elapsed = time.time() - start

print(f"[NODE] Entrenamiento completado en {elapsed:.2f}s")

# Guardar resultado en /results (solo tiempo)
results_dir = "/results"
os.makedirs(results_dir, exist_ok=True)

csv_basename = os.path.basename(csv_file).split('.')[0]
result_file = os.path.join(results_dir, f"result_{csv_basename}.txt")

print(f"[NODE] Guardando resultado en: {result_file}")

# Guardar solo el tiempo
try:
    with open(result_file, "w") as f:
        f.write(f"{elapsed}")
    print(f"[NODE] Tiempo guardado correctamente")
except Exception as e:
    print(f"[NODE] ERROR al guardar tiempo: {e}")

# Leer el dataset COMPLETO para hacer predicciones
# CORRECCIÓN: Usar path absoluto correcto
full_dataset_path = "/fulldata/curse.csv"  # Nuevo path
print(f"[NODE] Buscando dataset completo en: {full_dataset_path}")

if os.path.exists(full_dataset_path):
    print(f"[NODE] Dataset completo encontrado, cargando...")
    df_full = pd.read_csv(full_dataset_path)
    print(f"[NODE] Dataset completo cargado: {len(df_full)} muestras")
    
    # Aplicar el mismo preprocesamiento que en master
    df_full[target_col] = df_full[target_col].astype(str).str.strip().map({
        'Completed': 1,
        'Not Completed': 0
    })
    df_full = df_full.dropna(subset=[target_col])
    
    categorical_cols = ['Gender', 'Education_Level', 'Employment_Status', 
                        'City', 'Device_Type', 'Internet_Connection_Quality', 
                        'Course_Level', 'Payment_Mode']
    df_full = pd.get_dummies(df_full, columns=categorical_cols, drop_first=False)
    
    irrelevant_cols = ['Student_ID', 'Name', 'Course_ID', 'Course_Name', 'Category',
                       'Enrollment_Date', 'Fee_Paid', 'Discount_Used']
    df_full = df_full.drop(columns=[c for c in irrelevant_cols if c in df_full.columns], errors='ignore')
    df_full = df_full.fillna(0)
    
    X_full = df_full.drop(target_col, axis=1).select_dtypes(include=[np.number])
    
    print(f"[NODE] Haciendo predicciones sobre {len(X_full)} muestras...")
    
    # Hacer predicciones sobre el dataset completo
    predictions = model.predict(X_full)
    
    # Guardar las predicciones
    pred_file = os.path.join(results_dir, f"predictions_{csv_basename}.npy")
    try:
        np.save(pred_file, predictions)
        print(f"[NODE] Predicciones guardadas en {pred_file}")
    except Exception as e:
        print(f"[NODE] ERROR al guardar predicciones: {e}")
else:
    print(f"[NODE] WARNING: No se encontró dataset completo en {full_dataset_path}")
    print(f"[NODE] Contenido de /: {os.listdir('/')}")

print("[NODE] Proceso finalizado")