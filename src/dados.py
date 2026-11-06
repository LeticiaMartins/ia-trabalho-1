import pandas as pd
from sklearn.model_selection import train_test_split


def carregar_vinhos(pasta):
    tinto = pd.read_csv(pasta / "winequality-red.csv", sep=";")
    branco = pd.read_csv(pasta / "winequality-white.csv", sep=";")
    tinto["tipo"] = "tinto"
    branco["tipo"] = "branco"
    df = pd.concat([tinto, branco], ignore_index=True)

    # quality é uma nota dada por avaliadores, não uma medida química
    X = df.drop(columns=["tipo", "quality"])
    y = df["tipo"]
    return X, y


def carregar_pinguins(pasta):
    df = pd.read_csv(pasta / "penguins.csv")

    # TODO: o grupo decide se remove ou preenche as linhas com valores faltando
    df = df.dropna()

    # year é o ano da coleta, não uma característica do animal
    X = df.drop(columns=["species", "year"])
    X = pd.get_dummies(X, columns=["island", "sex"], dtype=int)
    y = df["species"]
    return X, y


def dividir(X, y, teste=0.2, seed=42):
    # estratificado pra manter a proporção das classes no treino e no teste
    return train_test_split(X, y, test_size=teste, stratify=y, random_state=seed)
