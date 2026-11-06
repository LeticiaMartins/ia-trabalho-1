# IA - Trabalho 1

Análise comparativa de algoritmos de classificação (Inteligência Artificial, UFABC).

Comparamos uma Rede Neural (MLP) com um segundo classificador em duas bases de dados públicas.

## Integrantes

- Leticia Martins
- Ana Clara de Lima Saraiva RA:11202131234
- 

## Estrutura

```
data/          bases de dados (.csv), ver data/README.md
src/
  dados.py         leitura e divisão treino/teste
  modelos.py       definição dos modelos e grades de hiperparâmetros
  avaliacao.py     métricas e matriz de confusão
  experimentos.py  script principal: treina e avalia os modelos
  graficos.py      gera os gráficos e as tabelas do relatório
resultados/    tabelas e figuras geradas
relatorio/     relatório em LaTeX (abnTeX2)
```

## Como executar

Requer Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/experimentos.py
python src/graficos.py
```

As bases já estão em `data/` (fontes em `data/README.md`). A execução leva alguns minutos.

Em `resultados/` são gerados:

- `resumo.csv`: métricas no conjunto de teste e melhores hiperparâmetros de cada modelo
- `hiperparametros_<base>_<modelo>.csv`: F1 médio na validação cruzada para cada combinação testada
- `matriz_<base>_<modelo>.png`: matrizes de confusão
- `mlp_hiperparametros_<base>.png`: efeito dos hiperparâmetros da MLP

O `graficos.py` também gera as tabelas do relatório em `relatorio/tabelas/`.

Para gerar o PDF do relatório (precisa de TeX Live com abnTeX2). No Overleaf, é preciso enviar o repositório inteiro, porque as figuras são lidas de `resultados/`:

```bash
cd relatorio
latexmk -pdf relatorio.tex
```

## Decisões do projeto

As decisões de pré-processamento de cada base estão em `data/README.md`. As demais:

- **Segundo algoritmo:** Random Forest, um dos modelos clássicos vistos em aula. Costuma ter bom desempenho sem muito ajuste, o que dá uma boa referência para comparar com a MLP.
- **Divisão dos dados:** 80% treino e 20% teste, estratificada pela classe, com `random_state=42` em todo o projeto para que os resultados sejam reproduzíveis.
- **Escolha dos hiperparâmetros:** busca em grade (`GridSearchCV`) com validação cruzada de 5 folds, usando só o conjunto de treino. O conjunto de teste é usado uma única vez, na avaliação final.
- **Métrica para escolher os hiperparâmetros:** F1 com média macro. A base de vinhos é desbalanceada (75% brancos), e a acurácia esconderia os erros na classe menor.
- **Hiperparâmetros testados na MLP:** camadas ocultas (16), (64) e (64, 32); ativação ReLU e tanh; taxa de aprendizado 0,001 e 0,01; regularização `alpha` 0,0001 e 0,01. São 24 combinações. O limite de iterações é 1000.
- **Hiperparâmetros testados no Random Forest:** 100 e 300 árvores; profundidade máxima 10, 20 ou sem limite.
- **Padronização:** só na MLP, que é sensível à escala dos atributos. O Random Forest não precisa, porque divide os dados por limiares em cada atributo.

## Bibliotecas

Só usamos as permitidas no enunciado: numpy, pandas, scikit-learn, matplotlib e seaborn.

## Entrega

Arquivo `IA_Trabalho_nome1_nome2_nome3.zip` com o relatório em PDF, a pasta `src/`, este README e o `requirements.txt`.
