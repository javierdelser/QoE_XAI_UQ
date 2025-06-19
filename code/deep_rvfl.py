import copy
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
from scipy import stats
from time import time

from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score
from mapie.regression import MapieRegressor

from sklearn.model_selection import KFold, RandomizedSearchCV

from pre_process import feature_engineering
from utils import RVFL

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
# matplotlib.use('Agg')

# Config
seed = 42
n_iter = 20
mae_lst, smape_lst, coverage_lst,conformal_scores_lst = [], [], [], []
train_time, infer_time = [], []

# Load and process data
df = pd.read_csv(r"D:\TFM\code\SNESet-main\benchmark\datasets\domain_general\random_sample_5k.csv")
df_features, df_labels = feature_engineering(df)
y = df_labels.values.ravel()
print(df_features.head()) 
print(df_features.info())

# KFold config
cv = KFold(n_splits=5, shuffle=True, random_state=seed)

# Model base
model = RVFL(n_nodes=50, lam=1.0, w_random_range=[-1, 1], b_random_range=[0, 1],
             activation='relu', n_layer=2, random_state=seed)

# Hiperparameter search
param_dist = {
    "n_nodes": stats.randint(5, 100),
    "lam": stats.uniform(1e-3, 1),
    "w_random_range": [[-1, 1]],
    "b_random_range": [[0, 1]],
    "activation": ['relu', 'sigmoid', 'tribas', 'radbas', 'sine', 'hardlim'],
    "n_layer": stats.randint(1, 10),
}

random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                   scoring='neg_mean_absolute_error',
                                   cv=cv, random_state=seed, n_iter=n_iter, verbose=1)

print("Starting hyperparameter search...")
start = time()
random_search.fit(df_features.values, y) ### mal??? (nested CV) / pasarle solo el trozo train
print("RandomizedSearchCV done in %.2f seconds." % (time() - start))
print("Best parameters:", random_search.best_params_)

# Best model
model_best = RVFL(**random_search.best_params_, random_state=seed)

# CROSS VALIDATION 
for fold, (train_idx, val_idx) in enumerate(cv.split(df_features)):
    print(f"\nFold {fold + 1}")

    X_train, X_val = df_features.iloc[train_idx].values, df_features.iloc[val_idx].values
    y_train, y_val = y[train_idx], y[val_idx]

    start = time()
    model_best.fit(X_train, y_train)
    train_time.append(time() - start)

    mapie = MapieRegressor(estimator=model_best, method="plus", cv="split") # conformal
    mapie.fit(X_train, y_train)

    start = time()
    y_pred, y_pis = mapie.predict(X_val, alpha=0.1)
    infer_time.append(time() - start)

    mae = model_best.eval_mae(X_val, y_val)
    smape = model_best.eval_smape(X_val, y_val)
    coverage = regression_coverage_score(y_val, y_pis[:, 0], y_pis[:, 1])

    # Conformal scores
    conformal_scores = np.maximum(
    y_val - y_pis[:, 1],  # cuánto supera por arriba
    y_pis[:, 0] - y_val   # cuánto supera por abajo
    )
    mean_conformal_score = np.mean(conformal_scores)
    conformal_scores_lst.append(mean_conformal_score)

    mae_lst.append(mae)
    smape_lst.append(smape)
    coverage_lst.append(coverage)

    print(f" MAE: {mae:.4f} | SMAPE: {smape:.4f} | Coverage: {coverage:.2f}")
    print(f"Conformal Score medio (Fold {fold+1}): {mean_conformal_score:.4f}")

    # # Fold visualization
    # plt.figure(figsize=(10, 5))
    # plt.plot(np.arange(len(y_val)), y_val, label="True")
    # plt.plot(np.arange(len(y_val)), y_pred, label="Predicted", linestyle='--')
    # plt.fill_between(np.arange(len(y_val)), y_pis[:, 0].squeeze(), y_pis[:, 1].squeeze(), alpha=0.3, label="Prediction Interval")
    # plt.title(f"Fold {fold+1} - Predicción con MAPIE")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

# SHAP
    if smape < 100:
        best_smape = smape
        best_model = copy.deepcopy(model_best)
        best_X_train = X_train
        best_X_val = X_val  # para luego tomar una instancia
        best_feature_names = df_features.columns

instance = pd.DataFrame(best_X_val[0:1], columns=best_feature_names)
background = shap.sample(pd.DataFrame(best_X_train, columns=best_feature_names), 100)
explainer = shap.Explainer(lambda x: best_model.predict(x), background)
shap_values_instance = explainer(instance)
shap.plots.waterfall(shap_values_instance[0])
plt.show()
# plt.savefig(os.path.join(output_dir, "shap_waterfall.png"))
# plt.close()

X_val_df = pd.DataFrame(best_X_val[:10], columns=best_feature_names)  # <= usa varias filas
shap_values_summary = explainer(X_val_df)
shap.summary_plot(shap_values_summary.values, X_val_df, max_display=50)
plt.show()
# plt.savefig(os.path.join(output_dir, "shap_summary.png"))
# plt.close()

# --- ENTRENAMIENTO FINAL (para Mapie)---
print("\nTraining best model over the full dataset...")
model_best.fit(df_features.values, y)

mapie_final = MapieRegressor(estimator=model_best, method="plus", cv="split")
mapie_final.fit(df_features.values, y)
y_pred_final, y_pis_final = mapie_final.predict(df_features.values, alpha=0.1)

# --- PICKLE ---
output_dir = "D:\\TFM\\code\\SNESet-main\\benchmark\\outputs" 
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "model_best.pkl"), "wb") as f:
    pickle.dump(model_best, f)
with open(os.path.join(output_dir, "mapie_final.pkl"), "wb") as f:
    pickle.dump(mapie_final, f)

results = {
    "mae_lst": mae_lst,
    "smape_lst": smape_lst,
    "coverage_lst": coverage_lst,
    'conformal_scores': conformal_scores_lst,
    "train_time": train_time,
    "infer_time": infer_time
}
with open(os.path.join(output_dir, "rvfl_results.pkl"), "wb") as f:
    pickle.dump(results, f)
print(" Model, Mapie and results saved correctly!")

# with open("D:\\TFM\\code\\SNESet-main\\benchmark\\outputs\\model_best.pkl", "rb") as f:
#     model_best = pickle.load(f)


# --- VISUALIZACIONES GENERALES ---
idx = np.arange(len(y))

# 1. True vs Pred con intervalos
plt.figure(figsize=(14, 5))
plt.plot(idx, y, label="True")
plt.plot(idx, y_pred_final, label="Predicted", linestyle='--')
plt.fill_between(idx, y_pis_final[:, 0].squeeze(), y_pis_final[:, 1].squeeze(), alpha=0.3, label="Prediction Interval")
plt.title("Predicción general con MAPIE")
plt.xlabel("Muestras")
plt.ylabel("buffer_rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(output_dir, "general_prediction.png"))
# plt.close()

# 2. Histograma de errores
errors = y - y_pred_final
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, alpha=0.7)
plt.title("Distribución del error (True - Pred)")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Distribución del target
plt.figure(figsize=(8, 5))
plt.hist(y, bins=30, alpha=0.7)
plt.title("Distribución de buffer_rate")
plt.xlabel("buffer_rate")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Cobertura por fold
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(coverage_lst)+1), coverage_lst)
plt.axhline(0.9, color='red', linestyle='--', label="Objetivo de cobertura (90%)")
plt.xlabel("Fold")
plt.ylabel("Coverage")
plt.title("Cobertura por Fold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --- MÉTRICAS FINALES ---
print("\n Resultados generales:")
print(f"MAE medio (CV): {np.mean(mae_lst):.4f} ± {np.std(mae_lst):.4f}")
print(f"SMAPE medio (CV): {np.mean(smape_lst):.4f} ± {np.std(smape_lst):.4f}")
print(f"Coverage medio (CV): {np.mean(coverage_lst):.2f}")
print(f"Tiempo medio de entrenamiento: {np.mean(train_time):.2f} s")
print(f"Tiempo medio de inferencia: {np.mean(infer_time):.2f} s")
print(f"Conformal Score promedio: {np.mean(conformal_scores_lst)}± {np.std(conformal_scores_lst):.2f}")


################### RESULTADOS (5K) ##################
# Best parameters: {'activation': 'radbas', 'b_random_range': [0, 1], 'lam': 0.9395527090157502, 'n_layer': 2, 'n_nodes': 68, 'w_random_range': [-1, 1]}

# Fold 1
#  MAE: 0.0172 | SMAPE: 53.0113 | Coverage: 0.91
# Conformal Score medio (Fold 1): -0.0111

# Fold 2
#  MAE: 0.0163 | SMAPE: 49.7154 | Coverage: 0.85
# Conformal Score medio (Fold 2): -0.0077

# Fold 3
#  MAE: 0.0154 | SMAPE: 46.7649 | Coverage: 0.94
# Conformal Score medio (Fold 3): -0.0223

# Fold 4
#  MAE: 0.0165 | SMAPE: 48.0968 | Coverage: 0.89
# Conformal Score medio (Fold 4): -0.0129

# Fold 5
#  MAE: 0.0160 | SMAPE: 46.2177 | Coverage: 0.93
# Conformal Score medio (Fold 5): -0.0131
# PermutationExplainer explainer: 2it [00:48, 48.76s/it]
# PermutationExplainer explainer: 11it [00:10,  5.36s/it]

# Training best model over the full dataset...
#  Model, Mapie and results saved correctly!

#  Resultados generales:
# MAE medio (CV): 0.0163 ± 0.0006
# SMAPE medio (CV): 48.7612 ± 2.4442
# Coverage medio (CV): 0.90
# Tiempo medio de entrenamiento: 0.41 s
# Tiempo medio de inferencia: 0.05 s
# Conformal Score promedio: -0.013442894655520215± 0.00


