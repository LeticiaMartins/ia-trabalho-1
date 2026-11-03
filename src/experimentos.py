from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV

from dados import carregar_csv, dividir
from modelos import mlp, segundo_modelo
from avaliacao import metricas, salvar_matriz_confusao

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
RESULTADOS = RAIZ / "resultados"

# nome da base -> (arquivo, coluna alvo)
# TODO: preencher depois de escolher as bases
BASES = {
    "base1": ("base1.csv", "target"),
    "base2": ("base2.csv", "target"),
}

MODELOS = {
    "MLP": mlp,
    "RandomForest": segundo_modelo,
}


def rodar():
    RESULTADOS.mkdir(exist_ok=True)
    linhas = []

    for nome_base, (arquivo, alvo) in BASES.items():
        X, y = carregar_csv(DATA / arquivo, alvo)
        # TODO: pré-processamento específico de cada base (nulos, categóricas etc.)
        X_tr, X_te, y_tr, y_te = dividir(X, y)

        for nome_modelo, fabrica in MODELOS.items():
            modelo, grade = fabrica()
            busca = GridSearchCV(modelo, grade, cv=5, scoring="f1_macro", n_jobs=-1)
            busca.fit(X_tr, y_tr)

            # todas as combinações testadas, para a análise de hiperparâmetros
            pd.DataFrame(busca.cv_results_).to_csv(
                RESULTADOS / f"grid_{nome_base}_{nome_modelo}.csv", index=False
            )

            y_pred = busca.predict(X_te)
            res = metricas(y_te, y_pred)
            res.update(base=nome_base, modelo=nome_modelo, params=busca.best_params_)
            linhas.append(res)

            salvar_matriz_confusao(
                y_te, y_pred,
                f"{nome_modelo} - {nome_base}",
                RESULTADOS / f"cm_{nome_base}_{nome_modelo}.png",
            )
            print(f"{nome_base} | {nome_modelo} | f1={res['f1']:.3f}")

    pd.DataFrame(linhas).to_csv(RESULTADOS / "resumo.csv", index=False)


if __name__ == "__main__":
    rodar()
