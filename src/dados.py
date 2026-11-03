import pandas as pd
from sklearn.model_selection import train_test_split


def carregar_csv(caminho, coluna_alvo):
    df = pd.read_csv(caminho)
    X = df.drop(columns=[coluna_alvo])
    y = df[coluna_alvo]
    return X, y


def dividir(X, y, teste=0.2, seed=42):
    # estratificado pra manter a proporção das classes no treino e no teste
    return train_test_split(X, y, test_size=teste, stratify=y, random_state=seed)
