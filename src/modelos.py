from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# O scaler fica dentro do pipeline para ser ajustado só com os dados de treino
# em cada fold do GridSearch (evita data leakage).


def mlp(seed=42):
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(max_iter=1000, random_state=seed)),
    ])
    grade = {
        "clf__hidden_layer_sizes": [(16,), (64,), (64, 32)],
        "clf__activation": ["relu", "tanh"],
        "clf__learning_rate_init": [0.001, 0.01],
        "clf__alpha": [0.0001, 0.01],
    }
    return modelo, grade


# TODO: confirmar com o grupo qual vai ser o segundo algoritmo
def segundo_modelo(seed=42):
    modelo = Pipeline([
        ("clf", RandomForestClassifier(random_state=seed)),
    ])
    grade = {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 10, 20],
    }
    return modelo, grade
