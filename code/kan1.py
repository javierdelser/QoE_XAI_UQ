import pandas as pd
import torch
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy import stats
from time import time
from utils import *
from pre_process import *
import pickle

seed = 42
n_iter = 10
n_epochs = 100
batch_size = 64
lr = 0.01
mae_lst, smape_lst, train_time, infer_time = [], [], [], []

# Load dataset
df = pd.read_csv(r"D:\\TFM\\code\\SNESet-main\\benchmark\\datasets\\domain_general\\random_sample_5k.csv")
df_features, df_labels = feature_engineering(df)
input_dim = df_features.shape[1] # len(df_features.columns)
print("Total of features: ", input_dim)

cv = KFold(n_splits=5, shuffle=True, random_state=seed)
model = KANWrapper(width=[input_dim,5,1], k=3, seed=seed, lr=lr, epochs=n_epochs, batch_size=batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

param_dist = {
    "lr": stats.loguniform(1e-4, 1e-1), # tasa aprendizaje
    "width": [ # capas ocultas
        [input_dim, 16, 1],  # profundidad 1 oculta
        [input_dim, 32, 1],
        [input_dim, 64, 1],
        [input_dim, 128, 1],
        [input_dim, 64, 64, 1],  # profundidad 2
    ],
    "k": [2, 3],  # grado de splines (lineal/cuadratico/cubico)
    "batch_size": [32, 64],
}


random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    scoring='neg_mean_absolute_error',
    cv=cv,
    random_state=seed,
    verbose=2
)

start = time()
random_search.fit(df_features.values, df_labels.values)
print("RandomizedSearchCV took %.2f seconds for %d iterations." % ((time() - start), n_iter))
print("Best parameters: ", random_search.best_params_)
model_best = KANWrapper(
    width=random_search.best_params_["width"],
    k=random_search.best_params_["k"],
    lr=random_search.best_params_["lr"],
    batch_size=random_search.best_params_["batch_size"],
    seed=seed,
    epochs=n_epochs
)

for fold, (train_idx, test_idx) in enumerate(cv.split(df_features)):
    print(f"\nFold {fold + 1}/5")
    
    X_train, X_test = df_features.iloc[train_idx, :], df_features.iloc[test_idx, :]
    y_train, y_test = df_labels.iloc[train_idx].values, df_labels.iloc[test_idx].values

    model_best.fit(X_train.values, y_train)

    y_pred = model_best.predict(X_test.values)

    y_true = y_test.flatten()
    mae_val = mae(y_true, y_pred)
    smape_val = smape(y_true, y_pred)

    print(f"MAE: {mae_val:.5f}, SMAPE: {smape_val:.5f}")
    mae_lst.append(mae_val)
    smape_lst.append(smape_val)


# Resultados finales
print("\nResultados MAE por pliegue:", mae_lst)
print("MAE promedio:", np.mean(mae_lst))
print("Desviación estándar del MAE:", np.std(mae_lst))

print("Resultados SMAPE por pliegue:", smape_lst)
print("SMAPE promedio:", np.mean(smape_lst))
print("Desviación estándar del SMAPE:", np.std(smape_lst))

# --- GUARDAR CON PICKLE ---
output_dir = "D:\\TFM\\code\\SNESet-main\\benchmark\\outputs"
os.makedirs(output_dir, exist_ok=True)

# Guardar modelo KAN mejor encontrado
with open(os.path.join(output_dir, "model_best_KAN.pkl"), "wb") as f:
    pickle.dump(model_best, f)

# Guardar resultados de la búsqueda aleatoria
with open(os.path.join(output_dir, "random_search_results_KAN.pkl"), "wb") as f:
    pickle.dump(random_search.cv_results_, f)

# Guardar métricas de validación cruzada
results = {
    "mae_lst": mae_lst,
    "smape_lst": smape_lst,
    "train_time": train_time,
    "infer_time": infer_time
}
with open(os.path.join(output_dir, "kan_results.pkl"), "wb") as f:
    pickle.dump(results, f)

print("Model, Random Search results, and cross-validation metrics for KAN saved correctly!")

# with open("best_model.pkl", "rb") as f:
#     model = pickle.load(f)

############# RESULTADOS (5K) #############

# [CV] END batch_size=64, k=2, lr=0.08906204386161681, width=[146, 16, 1]; total time=13.6min
# checkpoint directory created: ./model
# saving model version 0.0
# checkpoint directory created: ./model
# saving model version 0.0
# RandomizedSearchCV took 74953.97 seconds for 10 iterations.
# Best parameters:  {'batch_size': 64, 'k': 2, 'lr': 0.006173770394704572, 'width': [146, 32, 1]}
# checkpoint directory created: ./model
# saving model version 0.0

# Fold 1/5
# MAE: 0.01605, SMAPE: 49.53518
# Fold 2/5
# MAE: 0.01634, SMAPE: 49.98291
# Fold 3/5
# MAE: 0.01577, SMAPE: 46.66613
# Fold 4/5
# MAE: 0.01663, SMAPE: 47.67415
# Fold 5/5
# MAE: 0.01570, SMAPE: 44.46991

# Resultados MAE por pliegue: [0.016045814060498913, 0.016338622192614474, 0.015766585762163512, 0.01663062981717008, 0.015698365003357605]
# MAE promedio: 0.016096003367160917
# Desviación estándar del MAE: 0.00035038279891212596
# Resultados SMAPE por pliegue: [49.53517700914174, 49.98290589229487, 46.66612989241795, 47.67415229796648, 44.46991010551315]
# SMAPE promedio: 47.66565503946684
# Desviación estándar del SMAPE: 2.003829948668557


