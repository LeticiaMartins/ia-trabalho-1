from pathlib import Path
from ast import literal_eval

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

PASTA_RESULTADOS = Path(__file__).parent.parent / "resultados"
PASTA_TABELAS = Path(__file__).parent.parent / "relatorio" / "tabelas"

NOMES_BASES = {"vinhos": "Wine Quality", "pinguins": "Palmer Penguins"}
NOMES_MODELOS = {"MLP": "MLP", "RandomForest": "Random Forest"}


def numero(valor, casas=4):
    # vírgula como separador decimal, como pede a ABNT
    return f"{valor:.{casas}f}".replace(".", ",")


def numero_curto(valor):
    return str(valor).replace(".", ",")


def grafico_hiperparametros_mlp(nome_base):
    tabela = pd.read_csv(PASTA_RESULTADOS / f"hiperparametros_{nome_base}_MLP.csv")
    tabela = tabela.rename(columns={
        "mlp__hidden_layer_sizes": "Camadas ocultas",
        "mlp__activation": "Ativação",
        "mlp__learning_rate_init": "Taxa de aprendizado",
        "mlp__alpha": "Alpha",
    })
    tabela["Camadas ocultas"] = tabela["Camadas ocultas"].map(
        {"(16,)": "16", "(64,)": "64", "(64, 32)": "64 + 32"})
    tabela["Taxa de aprendizado"] = tabela["Taxa de aprendizado"].map(numero_curto)
    tabela["Alpha"] = tabela["Alpha"].map(numero_curto)

    fig, (esq, dir) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # cada ponto é a média do F1 sobre as combinações dos outros hiperparâmetros
    sns.pointplot(data=tabela, x="Camadas ocultas", y="f1_medio", hue="Ativação",
                  order=["16", "64", "64 + 32"], errorbar=None, dodge=0.2, ax=esq)
    sns.pointplot(data=tabela, x="Taxa de aprendizado", y="f1_medio", hue="Alpha",
                  hue_order=["0,0001", "0,01"], palette=["tab:green", "tab:purple"], errorbar=None, dodge=0.2, ax=dir)

    esq.set_title("(a) Arquitetura e função de ativação")
    dir.set_title("(b) Taxa de aprendizado e regularização")
    esq.set_ylabel("F1 médio na validação cruzada")
    dir.set_ylabel("")
    esq.yaxis.set_major_formatter(FuncFormatter(lambda v, _: numero(v, 4)))

    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / f"mlp_hiperparametros_{nome_base}.png", dpi=150)
    plt.close()


def tabela_desempenho():
    resumo = pd.read_csv(PASTA_RESULTADOS / "resumo.csv")

    linhas = []
    for _, linha in resumo.iterrows():
        linhas.append(
            f"{NOMES_BASES[linha['base']]} & {NOMES_MODELOS[linha['modelo']]} & "
            f"{numero(linha['acuracia'])} & {numero(linha['precisao'])} & "
            f"{numero(linha['recall'])} & {numero(linha['f1'])} \\\\"
        )

    texto = "\n".join([
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Base & Modelo & Acurácia & Precisão & Revocação & F1 \\",
        r"\midrule",
        *linhas,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    (PASTA_TABELAS / "desempenho.tex").write_text(texto + "\n")


def tabela_melhores_hiperparametros():
    resumo = pd.read_csv(PASTA_RESULTADOS / "resumo.csv")

    linhas = []
    for _, linha in resumo.iterrows():
        parametros = literal_eval(linha["melhores_hiperparametros"])
        descricao = ", ".join(
            f"{nome.replace('mlp__', '')} = {valor}" for nome, valor in parametros.items()
        )
        descricao = descricao.replace("_", r"\_")
        linhas.append(
            f"{NOMES_BASES[linha['base']]} & {NOMES_MODELOS[linha['modelo']]} & "
            f"\\texttt{{{descricao}}} \\\\"
        )

    texto = "\n".join([
        r"\begin{tabular}{llp{8cm}}",
        r"\toprule",
        r"Base & Modelo & Melhor combinação \\",
        r"\midrule",
        "\n\\addlinespace\n".join(linhas),
        r"\bottomrule",
        r"\end{tabular}",
    ])
    (PASTA_TABELAS / "melhores_hiperparametros.tex").write_text(texto + "\n")


PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
for nome_base in NOMES_BASES:
    grafico_hiperparametros_mlp(nome_base)
tabela_desempenho()
tabela_melhores_hiperparametros()
print("Gráficos salvos em resultados/ e tabelas em relatorio/tabelas/")
