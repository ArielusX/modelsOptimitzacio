import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time
import sys
import os

# Argumento: CSV chunk
csv_file = sys.argv[1]

# Leer CSV
df = pd.read_csv(csv_file)

# Target
target_col = 'Completed'
if df[target_col].dtype == 'object':
    df[target_col] = df[target_col].map({'Completed':1, 'Not Completed':0})

# Seleccionar solo features numéricas
X = df.drop(target_col, axis=1).select_dtypes(include=[np.number])
y = df[target_col]

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

# Guardar resultado en /results
results_dir = "/results"
os.makedirs(results_dir, exist_ok=True)

csv_basename = os.path.basename(csv_file).split('.')[0]
result_file = os.path.join(results_dir, f"result_{csv_basename}.txt")

with open(result_file, "w") as f:
    f.write(str(elapsed))
