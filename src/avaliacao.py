import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def calcular_metricas(y_real, y_previsto):
    # média macro para não favorecer a classe majoritária
    return {
        "acuracia": accuracy_score(y_real, y_previsto),
        "precisao": precision_score(y_real, y_previsto, average="macro"),
        "recall": recall_score(y_real, y_previsto, average="macro"),
        "f1": f1_score(y_real, y_previsto, average="macro"),
    }


def salvar_matriz_confusao(y_real, y_previsto, titulo, arquivo):
    classes = sorted(y_real.unique())
    matriz = confusion_matrix(y_real, y_previsto, labels=classes)

    plt.figure(figsize=(5, 4))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.yticks(rotation=0)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(arquivo, dpi=150)
    plt.close()
