import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time

# ----------------- 1️⃣ Cargar y preparar dataset -----------------
df = pd.read_csv("curse.csv")

# Columnas categóricas seleccionadas
categorical_cols = [
    'Gender', 'Education_Level', 'Employment_Status', 
    'City', 'Device_Type', 'Internet_Connection_Quality', 
    'Course_Level', 'Payment_Mode'
]

# One-Hot Encoding


df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
irrelevant_cols = ['Student_ID', 'Name', 'Course_ID', 'Course_Name', 'Category', 'Enrollment_Date', 'Fee_Paid', 'Discount_Used']

# eliminar del dataset antes de separar X e y
df_encoded = df_encoded.drop(columns=irrelevant_cols, errors='ignore')

# Target: 'Completed' (codificar si es necesario)
if df_encoded['Completed'].dtype == 'object':
    le = LabelEncoder()
    df_encoded['Completed'] = le.fit_transform(df_encoded['Completed'])

X = df_encoded.drop('Completed', axis=1)
y = df_encoded['Completed']

print("Dimensiones del dataset final:", X.shape)

# ----------------- 2️⃣ Modelo -----------------
rf_params = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_leaf': 1,
    'n_jobs': 1  # 1 hilo por nodo
}

# ----------------- 3️⃣ Función entrenamiento nodo -----------------
def train_node(X_node, y_node, node_id):
    model = RandomForestClassifier(**rf_params)
    start = time.time()
    model.fit(X_node, y_node)
    t = time.time() - start
    print(f"Nodo {node_id} entrenado en {t:.2f} s")
    return t

# ----------------- 4️⃣ Función dividir dataset en N nodos equilibrados -----------------
def split_dataset(X, y, n_nodes):
    splits = np.array_split(np.arange(len(X)), n_nodes)
    X_nodes = [X.iloc[idx] for idx in splits]
    y_nodes = [y.iloc[idx] for idx in splits]
    return X_nodes, y_nodes

# ----------------- 5️⃣ Configurar número de nodos -----------------
n_nodes = 100  # por ejemplo, 4 nodos
X_nodes, y_nodes = split_dataset(X, y, n_nodes)

# ----------------- 6️⃣ Entrenar todos los nodos -----------------
times = []
for i in range(n_nodes):
    t = train_node(X_nodes[i], y_nodes[i], node_id=i+1)
    times.append(t)

makespan = max(times)
print(f"Tiempo total (makespan) para {n_nodes} nodos: {makespan:.2f} s")

# ----------------- 7️⃣ Guardar resultados en dataframe -----------------
benchmark_df = pd.DataFrame({
    'Nodo': list(range(1, n_nodes+1)),
    'n_rows': [len(X_nodes[i]) for i in range(n_nodes)],
    'n_cols': [X.shape[1]]*n_nodes,
    'n_estimators': [rf_params['n_estimators']]*n_nodes,
    'max_depth': [rf_params['max_depth']]*n_nodes,
    'tiempo_seg': times
})

print(benchmark_df)
benchmark_df.to_csv("benchmark_nodos.csv", index=False)
