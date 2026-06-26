# Próximos Passos (Next Steps)

Este guia serve para lembrar o que deve ser feito a seguir no repositório **predict-traffic-volume** para continuar o desenvolvimento do projeto final da USP.

---

## 1. Executar e Validar a Pipeline (O que foi implementado)

Como você ainda não executou a pipeline de pré-processamento e engenharia de atributos, siga estes passos para gerar a base de modelagem:

1.  **Ativar o ambiente virtual (Poetry)**:
    ```bash
    poetry shell
    ```
2.  **Rodar a pipeline por CLI**:
    Execute o comando configurado no `taskipy` para pré-processar os dados brutos e extrair as features:
    ```bash
    task preprocess
    ```
    *Isso gerará os arquivos `data/processed/clean_traffic_data.csv` e `models/scalers_and_encoders.pkl`.*
3.  **Executar e Inspecionar o Notebook interativo**:
    Abra e rode o notebook [`notebooks/02-preprocessing_and_features.ipynb`](file:///home/gabyl/projetos/predict-traffic-volume/notebooks/02-preprocessing_and_features.ipynb) para visualizar de forma interativa os gráficos de validação (remoção de outliers de $0\text{ K}$ e $9831\text{ mm}$ de chuva, além do tratamento das falhas silenciosas do sensor de neve).

---

## 2. Desenvolver as Próximas Fases da Disciplina

Uma vez gerada a base limpa, as próximas etapas a serem desenvolvidas são:

### A. Seleção de Atributos (Feature Selection)
*   Analisar a correlação das features cíclicas (`hour_sin`, `hour_cos`, `day_sin`, `day_cos`) e das novas flags climáticas (`is_raining`, `is_snowing`, `is_foggy_misty`) com o alvo `traffic_volume`.
*   Decidir quais atributos manter para simplificar o modelo e evitar multicolinearidade (por exemplo, decidir se usará o One-Hot Encoding de `weather_main` ou se focará apenas nas flags de descrição ou em representações simplificadas).

### B. Treinamento e Avaliação de Modelos (Regressão)
*   Criar o script de treinamento em `src/models/train_model.py`.
*   Implementar pelo menos **2 modelos distintos** conforme exigido pela disciplina (ex: Regressão Linear Baseline, Random Forest/Gradient Boosting e MLP).
*   Utilizar **Validação Cruzada K-Fold (k=5 ou k=10)** para garantir a robustez na avaliação.
*   Otimizar hiperparâmetros (via GridSearchCV ou RandomizedSearchCV).
*   Computar as métricas de regressão requisitadas: RMSE, MAE e $R^2$.

### C. Relatório Final e Apresentação
*   Redigir o relatório final em português (seções de Introdução, Trabalhos Relacionados, Materiais e Métodos, Experimentos e Resultados, e Conclusão).
*   Criar os slides da apresentação com base nos resultados obtidos.