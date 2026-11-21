# Manutenção Preditiva com Machine Learning

Projeto desenvolvido para o Módulo 1 para prever quando uma máquina vai quebrar na fábrica usando dados de sensores.

## Problema

Em uma fábrica monitorada por sensores, quando uma máquina quebra de surpresa, a produção para e gera prejuízo. O objetivo deste projeto é treinar modelos de Machine Learning para prever se a máquina vai falhar (`1`) ou continuar funcionando normal (`0`), ajudando a equipe de manutenção a agir antes da quebra.

## Base de dados

Usei o arquivo `data/manutencao_preditiva.csv`, que tem **10.000 linhas** e **14 colunas**, com leituras de temperatura do ar, temperatura do processo, velocidade de rotação (rpm), torque (Nm), desgaste da ferramenta (min), tipo do produto (`L`, `M`, `H`) e a coluna alvo `falha_maquina` (9.661 casos normais e 339 falhas).

As colunas `udi` e `id_produto` foram retiradas por serem apenas códigos de identificação. As colunas com o tipo específico de falha (`falha_twf`, `falha_hdf`, `falha_pwf`, `falha_osf` e `falha_rnf`) também saíram porque só existem depois que a máquina já quebrou, o que causaria vazamento de dados e foi proibido pela Engenharia.

## Técnicas e tecnologias

* **Linguagem:** Python 3.10

* **Bibliotecas:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn e Imbalanced-Learn

### Etapas aplicadas no Pipeline

1. Análise Exploratória (EDA) com dimensões, `.describe()`, gráficos de distribuição, contagem de falhas e correlação de Pearson.

2. Limpeza de dados (verificação de duplicadas, preenchimento de 2.000 nulos pela mediana por causa da assimetria de `velocidade_rotacao_rpm` e análise de outliers com boxplot e IQR).

3. Feature Engineering criando a coluna `potencia` (`velocidade_rotacao_rpm * torque_nm`).

4. Codificação da coluna `tipo` (`L = 0`, `M = 1`, `H = 2`), divisão 80% treino / 20% teste com `stratify=y` e balanceamento com `RandomUnderSampler` apenas no treino.

5. Escalonamento com `StandardScaler` apenas nas 6 colunas contínuas do KNN (`fit_transform` no treino e `transform` no teste), deixando a Árvore de Decisão sem escala.

6. Ajuste de parâmetros (`K = 3, 5 e 7` no KNN e `max_depth = 3, 5 e None` na Árvore) para evitar overfitting.

7. Avaliação final nos dados de teste com acurácia, matriz de confusão e escolha do melhor modelo.

## Estrutura do repositório

```text
manutencao_preditiva/
├── data/
│   └── manutencao_preditiva.csv
├── pipeline_preditivo.ipynb
├── README.md
└── requirements.txt
```

## Como executar

### Pré-requisitos

* **Python 3.10** (versão usada no projeto)

* **Git** instalado

### Passo a passo

1. Clone o repositório e entre na pasta:

```bash
git clone https://github.com/rafhaelcinelli/manutencao_preditiva.git
cd manutencao_preditiva
```

2. Crie e ative o ambiente virtual:

```bash
python -m venv .venv
# No Windows:
.venv\Scripts\activate
# No Linux/macOS:
source .venv/bin/activate
```

3. Instale as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

4. Abra o arquivo `pipeline_preditivo.ipynb` no VS Code, selecione o kernel do ambiente `.venv` e clique em **Run All**.

## Resultados

Avaliando os dois melhores modelos nos **2.000 registros de teste** (com 68 falhas reais e 1.932 operações normais):

| Modelo | Melhor Configuração | Acurácia no Treino | Acurácia no Teste | Falhas Detectadas (de 68) | Alarmes Falsos |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Árvore de Decisão** | `max_depth = 3` | 86,90% | **88,85%** | 53 | **208** |
| **KNN** | `K = 5` | 88,75% | **85,00%** | 55 | 287 |

### Veredito Final

A empresa deve adotar a **Árvore de Decisão com `max_depth = 3`**. 

Além de ganhar do KNN por **3,85 pontos percentuais** no teste sem sofrer overfitting (gerando **79 inspeções desnecessárias a menos** que o KNN), ela não precisa de escalonamento em produção e gera regras fáceis para os mecânicos e engenheiros entenderem quando a máquina vai quebrar.

## Melhorias futuras

1. O `RandomUnderSampler` descartou **7.458 registros** normais do treino; no futuro vale testar **SMOTENC** (que funciona bem com colunas categóricas como `tipo`) ou `class_weight='balanced'` para aproveitar toda a base.

2. Os parâmetros (`K` e `max_depth`) foram escolhidos olhando o teste; usar **validação cruzada (*Cross-Validation*)** dentro do treino deixaria a escolha mais rigorosa.

3. Testar outros valores de `K` e de profundidade da árvore, além de modelos como **Random Forest**.

## Organização do Git

Criei uma branch para cada fase, saindo da `develop`, e juntei tudo na `main` no final:

| Branch | Objetivo |
| :--- | :--- |
| `main` | Versão final consolidada do projeto |
| `develop` | Integração das fases durante o desenvolvimento |
| `feature/eda` | Fase 1: análise exploratória dos dados e gráficos |
| `feature/data-prep` | Fase 2: duplicatas, preenchimento pela mediana e outliers |
| `feature/engenharia-atributos` | Fase 3: criação da variável `potencia` |
| `feature/divisao-balanceamento` | Fase 4: codificação do `tipo`, divisão 80/20 e undersampling |
| `feature/escalonamento` | Fase 5: aplicação do `StandardScaler` no KNN |
| `feature/ajuste-modelos` | Fase 6: variação de `K` e `max_depth` e análise de overfitting |
| `feature/avaliacao-final` | Fase 7: avaliação final no teste e conclusão |
| `feature/documentacao` | Criação do `requirements.txt` e do `README.md` |
| `feature/ajuste-readme` | Revisão dos textos do notebook, correção da formatação do README e inclusão do link do vídeo. |

## Vídeo

* Link da apresentação no Google Drive: 