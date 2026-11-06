from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV

from dados import carregar_vinhos, carregar_pinguins, dividir
from modelos import criar_mlp, criar_random_forest
from avaliacao import calcular_metricas, salvar_matriz_confusao

PASTA_DADOS = Path(__file__).parent.parent / "data"
PASTA_RESULTADOS = Path(__file__).parent.parent / "resultados"

bases = [
    ("vinhos", carregar_vinhos),
    ("pinguins", carregar_pinguins),
]

modelos = [
    ("MLP", criar_mlp),
    ("RandomForest", criar_random_forest),
]

PASTA_RESULTADOS.mkdir(exist_ok=True)
resumo = []

for nome_base, carregar in bases:
    X, y = carregar(PASTA_DADOS)
    X_treino, X_teste, y_treino, y_teste = dividir(X, y)

    for nome_modelo, criar in modelos:
        modelo, hiperparametros = criar()

        # testa todas as combinações com validação cruzada de 5 folds no treino
        busca = GridSearchCV(modelo, hiperparametros, cv=5, scoring="f1_macro", n_jobs=-1)
        busca.fit(X_treino, y_treino)

        tabela = pd.DataFrame(busca.cv_results_["params"])
        tabela["f1_medio"] = busca.cv_results_["mean_test_score"]
        tabela["desvio"] = busca.cv_results_["std_test_score"]
        tabela = tabela.sort_values("f1_medio", ascending=False).round(4)
        tabela.to_csv(PASTA_RESULTADOS / f"hiperparametros_{nome_base}_{nome_modelo}.csv", index=False)

        # avaliação final no conjunto de teste, que não foi usado até aqui
        y_previsto = busca.predict(X_teste)
        metricas = calcular_metricas(y_teste, y_previsto)
        salvar_matriz_confusao(
            y_teste, y_previsto,
            f"{nome_modelo} - {nome_base}",
            PASTA_RESULTADOS / f"matriz_{nome_base}_{nome_modelo}.png",
        )

        resumo.append({"base": nome_base, "modelo": nome_modelo, **metricas,
                       "melhores_hiperparametros": busca.best_params_})
        print(f"{nome_base} | {nome_modelo} | F1 = {metricas['f1']:.3f}")

pd.DataFrame(resumo).round(4).to_csv(PASTA_RESULTADOS / "resumo.csv", index=False)
