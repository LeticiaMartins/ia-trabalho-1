# Bases de dados

| Base | Fonte | Arquivos | Alvo | Licença |
|------|-------|----------|------|---------|
| Wine Quality | [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality) | `winequality-red.csv`, `winequality-white.csv` | tipo do vinho (tinto ou branco) | CC BY 4.0 |
| Palmer Penguins | [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/) | `penguins.csv` | espécie (Adelie, Chinstrap, Gentoo) | CC0 |

Os arquivos já estão nesta pasta, sem nenhuma alteração em relação à fonte. As mudanças abaixo são feitas pelo código, em `src/dados.py`, no momento da leitura.

## Pré-processamento

### Wine Quality

- Os dois arquivos (1.599 tintos e 4.898 brancos) são unidos em uma base de 6.497 vinhos.
- É criada a coluna alvo `tipo`, com os valores `tinto` e `branco`, a partir do arquivo de origem de cada vinho.
- A coluna `quality` é removida. Ela é uma nota dada por avaliadores, não uma medida química do vinho.
- A base não tem valores faltando.

Ficam 11 atributos numéricos. A proporção é de cerca de 25% tintos e 75% brancos.

### Palmer Penguins

- Das 344 linhas, 11 têm algum valor faltando: 2 sem nenhuma medida e outras 9 sem o sexo. Essas 11 linhas são removidas, ficando 333 pinguins (146 Adelie, 119 Gentoo e 68 Chinstrap).
- A coluna `year` é removida. Ela é o ano da coleta, não uma característica do animal.
- As colunas de texto `island` e `sex` viram colunas de 0 e 1, uma para cada valor (one-hot).

Ficam 4 medidas numéricas e 5 colunas de 0 e 1.

### Etapas comuns às duas bases

- Divisão em 80% treino e 20% teste, estratificada pela classe, com `random_state=42`.
- Na MLP, os atributos são padronizados com `StandardScaler`. O scaler fica dentro do pipeline, então é ajustado só com os dados de treino de cada fold.

Referências:

- CORTEZ, P. et al. Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, v. 47, n. 4, p. 547-553, 2009.
- HORST, A. M.; HILL, A. P.; GORMAN, K. B. palmerpenguins: Palmer Archipelago (Antarctica) penguin data. R package, 2020.
