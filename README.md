Manutenção Preditiva 4.0 — Detecção de Falhas Mecânicas

Sistema de Inteligência Artificial aplicado à Indústria 4.0 para prever falhas mecânicas em maquinário industrial a partir da leitura de sensores operacionais, permitindo intervenções preventivas antes que ocorram paradas não programadas na linha de produção.

Problema

Em parques fabris monitorados por sensores, quebras inesperadas de equipamentos geram prejuízos financeiros, atrasos na produção e riscos operacionais. O desafio deste projeto é construir um pipeline completo de Ciência de Dados capaz de classificar o estado de funcionamento das máquinas em duas categorias:
`0` (Normal): Operação regular sem risco imediato de avaria.
`1` (Falha): Condição crítica que indica risco de quebra mecânica (`falha_maquina`).
Como apenas 3,39% dos registros históricos representam falhas reais, o sistema precisa lidar com o forte desbalanceamento dos dados e evitar o sobreajuste (overfitting) para ser confiável em ambiente real.

Base de dados

A base utilizada (`data/manutencao_preditiva.csv`) contém 10.000 registros e 14 colunas com leituras de sensores e metadados de operação:
Variáveis preditoras utilizadas: `tipo` (qualidade do produto: `L`, `M`, `H`, codificada em `0, 1, 2`), `temperatura_ar_k`, `temperatura_processo_k`, `velocidade_rotacao_rpm`, `torque_nm`, `desgaste_ferramenta_min` e a variável criada `potencia`.
Variáveis descartadas: Identificadores (`udi`, `id_produto`) por não terem valor preditivo, e subtipos de falha (`falha_twf`, `falha_hdf`, `falha_pwf`, `falha_osf`, `falha_rnf`) por representarem diagnósticos posteriores à quebra (o que causaria Data Leakage).
Variável alvo: `falha_maquina` (9.661 casos normais e 339 falhas).

Técnicas e tecnologias

Linguagem: Python 3.10
Manipulação e Análise de Dados: `pandas`, `numpy`
Visualização de Dados: `matplotlib`, `seaborn`
Machine Learning e Pré-processamento: `scikit-learn` (`train_test_split`, `StandardScaler`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `accuracy_score`, `confusion_matrix`, `classification_report`)
Balanceamento de Dados: `imbalanced-learn` (`RandomUnderSampler`)

Etapas aplicadas no Pipeline:

Análise Exploratória (EDA) com estatísticas descritivas e gráficos de distribuição/correlação.
Limpeza de dados (verificação de duplicadas, imputação de 2.000 valores nulos pela mediana devido à assimetria de `velocidade_rotacao_rpm` e inspeção de outliers via regra do IQR).
Engenharia de Recursos (Feature Engineering) com criação da variável `potencia = velocidade_rotacao_rpm * torque_nm`.
Divisão estratificada 80/20 (`stratify=y`) e balanceamento com `RandomUnderSampler` aplicado exclusivamente no treino.
Escalonamento (`StandardScaler`) aplicado apenas nas 6 variáveis contínuas do KNN (`fit_transform` no treino, `transform` no teste), mantendo a Árvore de Decisão sem escalonamento.
Ajuste de hiperparâmetros ($K \in {3, 5, 7}$ no KNN e $\text{max_depth} \in {3, 5, \text{None}}$ na Árvore) para controle de overfitting.
Avaliação final dos melhores modelos no teste, com cálculo de acurácia, matriz de confusão e veredito para implantação.
Estrutura do repositório
```text
├── data/
│   └── manutencao_preditiva.csv    # Base de dados histórica dos sensores
├── pipeline_preditivo.ipynb        # Notebook principal segmentado nas 7 fases
├── requirements.txt                # Dependências e bibliotecas do projeto
└── README.md                       # Documentação completa da solução
```
Como executar
Pré-requisitos
Python 3.10 (versão utilizada no desenvolvimento)
Git instalado
Passo a passo
Clone este repositório para sua máquina local:
```bash
   git clone https://github.com/rafhaelcinelli/manutencao_preditiva.git
   cd manutencao_preditiva
   ```
Crie e ative um ambiente virtual:
```bash
   python -m venv .venv
   # No Windows:
   .venv\Scripts\activate
   # No Linux/macOS:
   source .venv/bin/activate
   ```
Instale as bibliotecas necessárias:
```bash
   pip install -r requirements.txt
   ```
Abra o `pipeline_preditivo.ipynb` no VS Code, selecione o kernel do ambiente virtual (`.venv`) e clique em Run All.

Resultados
Após o ajuste de hiperparâmetros na Fase 6, os dois melhores modelos foram avaliados sobre os 2.000 registros inéditos da base de teste (contendo 68 falhas reais e 1.932 operações normais):

Modelo	Melhor Configuração	Acurácia no Treino	Acurácia no Teste	Falhas Detectadas (de 68)	Alarmes Falsos (Falsos Positivos)
Árvore de Decisão	`max_depth = 3`	86,90%	88,85%	53	208
KNN	`n_neighbors = 5`	88,75%	85,00%	55	287

Veredito Final

O modelo selecionado para implantação é a Árvore de Decisão com `max_depth = 3`. Além de alcançar 88,85% de acurácia no teste (3,85 pontos percentuais acima do melhor KNN) sem apresentar overfitting (ao contrário de `max_depth = None`, que memorizou 100% do treino e caiu para 87,20% no teste), ela gera 79 inspeções desnecessárias a menos que o KNN, dispensa escalonamento em produção e fornece regras simples e auditáveis pela equipe de manutenção.

Melhorias futuras

Aproveitamento integral da base no balanceamento: Substituir o `RandomUnderSampler` (que descartou 7.458 registros de operação normal do treino) por técnicas como SMOTENC (variante própria para bases que misturam variáveis numéricas e categóricas como `tipo`) ou ajuste de pesos (`class_weight='balanced'`).
Validação Cruzada (Cross-Validation): Utilizar validação cruzada estratificada dentro do conjunto de treino para selecionar hiperparâmetros sem depender diretamente da avaliação no conjunto de teste.

Exploração de novos hiperparâmetros e algoritmos: Testar faixas maiores de $K$ e profundidade máxima, além de avaliar modelos baseados em comitês de árvores como Random Forest.
Organização do Git
O versionamento deste projeto seguiu o fluxo de ramificações (Git Flow simplificado), mantendo todas as feature branches preservadas no histórico após os merges:

Branch	Objetivo
`main`	Versão final consolidada e estável do projeto para entrega.
`develop`	Branch de integração das funcionalidades desenvolvidas antes do merge final na `main`.
`feature/eda`	Fase 1: análise exploratória de dados (dimensões, tipos, resumo estatístico e gráficos).
`feature/data-prep`	Fase 2: remoção de duplicatas, imputação pela mediana e análise de outliers via IQR.
`feature/engenharia-atributos`	Fase 3: criação e validação da variável numérica `potencia`.
`feature/divisao-balanceamento`	Fase 4: codificação da variável `tipo`, divisão estratificada 80/20 e undersampling no treino.
`feature/escalonamento`	Fase 5: aplicação do `StandardScaler` nas variáveis contínuas para o KNN.
`feature/ajuste-modelos`	Fase 6: variação dos hiperparâmetros ($K$ e `max_depth`) e análise de overfitting.
`feature/avaliacao-final`	Fase 7: avaliação final das acurácias no teste, matrizes de confusão e veredito.
`feature/documentacao`	Criação do arquivo de dependências `requirements.txt` e da documentação no `README.md`.

Vídeo
Link da apresentação em vídeo (Google Drive): 