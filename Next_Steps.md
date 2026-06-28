# Próximos Passos (Next Steps)

Este guia serve para lembrar o estado atual do projeto **predict-traffic-volume** para continuar o desenvolvimento do projeto final da USP.

---

## 1. Revisão de EDA e Seleção de Features (Para Amanhã)

A pipeline de pré-processamento já foi executada e as novas features (tempo cíclico, flags de clima, flags de rush/weekend) foram geradas e validadas estatisticamente. 

**Tarefa Imediata do Usuário:**
Ler o documento de insights `notebooks/02-eda_after_preprocess_insights.md` ou abrir interativamente o notebook `notebooks/02-eda_after_preprocessing.ipynb` para refletir sobre os achados e aprovar formalmente as decisões de "Feature Selection". 
*   **Foco principal:** Entender por que vamos excluir as variáveis originais de data (`hour`, `month`, etc) e por que vamos excluir a string original `weather_main` em favor das flags booleanas criadas, evitando a *multicolinearidade*.

---

## 2. Preparação dos Dados para Modelagem (Data Splitting & Scaling)

Após a aprovação formal dos insights do passo 1, a Antigravity (IA) irá:
*   Remover do dataset as colunas decididas na etapa de Feature Selection (listadas em `02-eda_after_preprocess_insights.md`).
*   Dividir os dados em **Treino, Validação (Dev) e Teste**.
*   Ajustar os **Scalers** (como o StandardScaler para variáveis numéricas como `temp` e `clouds_all`) **estritamente no conjunto de Treinamento** para evitar *Data Leakage*.
*   Aplicar a transformação nos conjuntos de Validação e Teste.

---

## 3. Treinamento e Avaliação de Modelos de Regressão

Com os dados blindados contra vazamentos e perfeitamente particionados, iniciaremos a criação matemática dos modelos em `src/models/train_model.py`:
*   Implementar pelo menos **2 modelos distintos** conforme exigido pela disciplina (ex: Regressão Linear Baseline, Random Forest/Gradient Boosting, e uma Rede Neural Simples / MLP).
*   Utilizar **Validação Cruzada K-Fold (k=5 ou k=10)** para garantir a robustez.
*   Otimizar os hiperparâmetros (via GridSearchCV ou RandomizedSearchCV).
*   Computar e comparar as métricas de regressão: **RMSE, MAE e $R^2$**.

---

## 4. Entregáveis Finais (Relatório e Apresentação)

*   Redigir o relatório final **em português** detalhando as metodologias do repositório (Introdução, Trabalhos Relacionados, Materiais e Métodos, Experimentos e Resultados, e Conclusão).
*   Criar os slides resumo da apresentação com foco nos insights e resultados obtidos.