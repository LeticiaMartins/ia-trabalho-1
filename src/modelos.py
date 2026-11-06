from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def criar_mlp():
    # o scaler fica no pipeline para ser ajustado só com o treino (sem data leakage)
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(max_iter=1000, random_state=42)),
    ])
    hiperparametros = {
        "mlp__hidden_layer_sizes": [(16,), (64,), (64, 32)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__learning_rate_init": [0.001, 0.01],
        "mlp__alpha": [0.0001, 0.01],
    }
    return modelo, hiperparametros


def criar_random_forest():
    modelo = RandomForestClassifier(random_state=42)
    hiperparametros = {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
    }
    return modelo, hiperparametros
