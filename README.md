# IA - Trabalho 1

Análise comparativa de algoritmos de classificação (Inteligência Artificial, UFABC).

Comparamos uma Rede Neural (MLP) com um segundo classificador em duas bases de dados públicas.

## Integrantes

- Leticia Martins
- 
- 

## Estrutura

```
data/          bases de dados (.csv), ver data/README.md
src/
  dados.py         leitura e divisão treino/teste
  modelos.py       definição dos modelos e grades de hiperparâmetros
  avaliacao.py     métricas e matriz de confusão
  experimentos.py  script principal
resultados/    tabelas e figuras geradas
relatorio/     relatório em LaTeX (abnTeX2), compila no Overleaf
```

## Como executar

Requer Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/experimentos.py
```

As bases já estão em `data/` (fontes em `data/README.md`). A execução leva alguns minutos.

Em `resultados/` são gerados:

- `resumo.csv`: métricas no conjunto de teste e melhores hiperparâmetros de cada modelo
- `hiperparametros_<base>_<modelo>.csv`: F1 médio na validação cruzada para cada combinação testada
- `matriz_<base>_<modelo>.png`: matrizes de confusão

Para gerar o PDF do relatório (precisa de TeX Live com abnTeX2, ou usar o Overleaf):

```bash
cd relatorio
latexmk -pdf relatorio.tex
```

## Bibliotecas

Só usamos as permitidas no enunciado: numpy, pandas, scikit-learn, matplotlib e seaborn.

## Entrega

Arquivo `IA_Trabalho_nome1_nome2_nome3.zip` com o relatório em PDF, a pasta `src/`, este README e o `requirements.txt`.
