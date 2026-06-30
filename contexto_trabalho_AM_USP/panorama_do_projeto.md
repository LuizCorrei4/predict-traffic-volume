# Panorama Geral do Projeto

Este documento apresenta um resumo de tudo o que foi realizado até o momento no projeto **predict-traffic-volume**, alinhado com as especificações exigidas para o trabalho final da disciplina SCC-276 (Aprendizado de Máquina - USP).

## 1. Organização do Repositório e Ferramentas (Concluído)
- **Gerenciamento de Dependências**: O projeto adotou o **Poetry** para gerenciar dependências e empacotamento, garantindo um ambiente isolado e reprodutível (`pyproject.toml` e `poetry.lock`).
- **Bibliotecas Configuradas**: Foram incluídas as principais bibliotecas para o desenvolvimento: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `optuna` e bibliotecas de visualização (`matplotlib`, `seaborn`).
- **Estrutura de Pastas**: Construída seguindo boas práticas de Ciência de Dados, isolando `/data`, `/src` (modelos e pipelines) e `/reports/logs`.

## 2. Exploração de Dados e Engenharia de Features (Concluído)
- Foram elaborados dois notebooks robustos de EDA (`01-exploration.ipynb` e `02-eda_after_preprocessing.ipynb`) que desvendaram lacunas severas nos dados (gaps de tempo), provando a incapacidade do uso de modelos ARIMA/SARIMA e validando o uso de algoritmos clássicos de ML.
- **Pré-processamento**: Transformação matemática do clima e uso de senos e cossenos (codificação cíclica) para as horas do dia, dias da semana e meses, permitindo que a IA entenda a natureza temporal.
- **Data Splitting Blindado**: Implementação de divisão estrita cronológica (Treino 70%, Validação 15%, Teste 15%). 
- Tratamento de nulos (`SimpleImputer`) e `StandardScaler` aplicados estritamente na partição de treino para evitar vazamento de dados (*data leakage*).

## 3. Implementação e Orquestração de Modelos (Concluído)
- **Fábrica de Algoritmos**: Script modular que instancia modelos de ponta (`Ridge`, `Random Forest`, `XGBoost`, `LightGBM`, `CatBoost`, `MLP`) com máximo aproveitamento de núcleos (CPU/GPU).
- **Avaliador e Logs**: Script autônomo que gera métricas (RMSE, MAE, R², MAPE, Max Error) e salva resultados rastreáveis em formato JSON.
- **O Maestro (Orquestrador)**: O script mestre (`train_model.py`) foi construído com uso nativo de otimização Bayesiana (`Optuna`) e validação cruzada (`TimeSeriesSplit`), garantindo a extração dos melhores hiperparâmetros sem espiar o conjunto de testes final.

## 4. Próximos Passos (Alinhados ao Roteiro do Projeto Final)
Com a fundação do projeto solidificada, o código em pé e rodando, as próximas etapas a serem desenvolvidas para cumprimento dos requisitos da disciplina são:

1. **Busca Profunda de Hiperparâmetros**: Executar o Optuna com mais *trials* (atualmente em andamento) para extrair o máximo de precisão dos modelos.
2. **Avaliação no Conjunto de Teste (The Vault)**: Após identificar o melhor algoritmo e seus hiperparâmetros na Validação, testar 1 vez no Conjunto de Teste para confirmar a acurácia final.
3. **Módulo de Plotagem Visual**: Construir gráficos elegantes que comparem a performance e o tempo de treinamento dos modelos.
4. **Redação do Relatório Final (Em elaboração)**:
   - **Introdução e Trabalhos Relacionados**: Explicar a contribuição frente a outras obras.
   - **Material e Métodos**: Descrever os modelos e as engenhosidades técnicas (Target Encoding, Target Scaling).
   - **Experimentos**: Compilar tabelas e figuras visuais comprobatórias.
   - **Conclusão**.
5. **Apresentação Final**: Criação dos slides (Resumo Executivo do Relatório Final).
