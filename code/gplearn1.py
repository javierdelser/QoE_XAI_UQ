from pre_process import feature_engineering, mae, smape
import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer
from scipy import stats
from time import time
import graphviz

# Config
seed = 42
n_iter = 10

# Load dataset
df = pd.read_csv(r"D:\TFM\code\SNESet-main\benchmark\datasets\domain_general\random_sample_5w.csv")
df_features, df_labels = feature_engineering(df)

# Define scorer functions
scorer_mae = make_scorer(mae, greater_is_better=False)
scorer_smape = make_scorer(smape, greater_is_better=False)

# CV strategy
cv = KFold(n_splits=2, shuffle=True, random_state=seed)

# Define model
base_model = SymbolicRegressor(
    population_size=1000,
    generations=10,
    stopping_criteria=0.01,
    function_set=('add', 'sub', 'mul', 'div'),
    parsimony_coefficient=0.0001,
    const_range=(-1.0, 1.0),
    init_depth=(2, 6),
    init_method='half and half',
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    random_state=seed,
    n_jobs=1
)

# Hyperparameter tuning
param_dist = {
    "parsimony_coefficient": stats.uniform(0.0001, 0.01),
    "const_range": [(-1.0, 1.0), (-10.0, 10.0)],
    "p_crossover": stats.uniform(0.5, 0.3),
    "p_subtree_mutation": stats.uniform(0.05, 0.15),
    "p_hoist_mutation": stats.uniform(0.01, 0.1),
    "p_point_mutation": stats.uniform(0.01, 0.1),
}

random_search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    scoring='neg_mean_absolute_error',
    n_iter=n_iter,
    cv=cv,
    random_state=seed
)

start = time()

random_search.fit(df_features.values, df_labels.values)
print("RandomizedSearchCV tardó %.2f segundos para %d iteraciones." % ((time() - start), n_iter))
print("Best parameters: ", random_search.best_params_)
model_best = random_search.best_estimator_

# Cross-Validation
print("Resultados CV con mejores hiperparámetros:")
scores_mae = cross_val_score(model_best, df_features.values, df_labels.values, scoring=scorer_mae, cv=cv)
scores_smape = cross_val_score(model_best, df_features.values, df_labels.values, scoring=scorer_smape, cv=cv)

print("Resultados MAE por pliegue:", -scores_mae)
print("MAE promedio:", -scores_mae.mean())
print("Desviación estándar del MAE:", scores_mae.std())

print("Resultados SMAPE por pliegue:", -scores_smape)
print("SMAPE promedio:", -scores_smape.mean())
print("Desviación estándar del SMAPE:", scores_smape.std())

# Fórmula obtenida
print("Mejor fórmula encontrada por GP:", model_best._program)
print("Mejor fórmula encontrada por GP:", str(model_best._program))
dot_string = model_best._program.export_graphviz()
graph = graphviz.Source(dot_string)  # Crear el objeto Graphviz
graph.render(filename="tree_visualization", format="pdf", view=True) 

# --- NOTAS ---
# Si hay overfitting -> subir parsimony_coefficient.
# Si va muy lento-> bajar population_size o generations.
# Para más diversidad de formulas-> usar function_set más amplio: sin, cos, log, sqrt, etc.

################# RESULTADOS ###################
# Resultados MAE por pliegue: [0.01835247 0.0180601  0.01891004]
# MAE promedio: 0.018440868807849065 ± 0.00035257173667554233
# Resultados SMAPE por pliegue: [53.52280927 58.86409337 56.05666744]
# SMAPE promedio: 56.147856692155074 ± 2.181523258833559
# Mejor fórmula encontrada por GP: X107
# Mejor fórmula encontrada por GP: X107