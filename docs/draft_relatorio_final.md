# Previsão de Volume de Tráfego: Uma Abordagem Comparativa com Algoritmos de Machine Learning

**Membros do Grupo:** [Nomes dos Membros]  
**Disciplina:** SCC-276 — Aprendizado de Máquina (USP)  
**Professora:** Roseli Aparecida Francelin Romero  

---

## I. Introdução
*(Descrever o impacto do trânsito na sociedade moderna, a importância de prever volumes de tráfego para logística e planejamento urbano. Introduzir o dataset da Interestadual I-94. Apresentar os modelos escolhidos: clássicos, bagging, boosting e redes neurais. Resumir o que o leitor encontrará nas seções seguintes.)*

## II. Trabalhos Relacionados
*(Aqui serão listados no mínimo 5 artigos recentes que aplicaram Machine Learning para previsão de tráfego ou uso de dados climáticos. O arquivo `docs/references.md` servirá como base de consulta para preencher esta seção. Ao final, destacar que nossa contribuição envolve o rigor metodológico de evitar Data Leakage com separação temporal estrita e o uso extensivo de Otimização Bayesiana.)*

## III. Material e Métodos

### A) Apresentação do Dataset
A base de dados escolhida foi a **Metro Interstate Traffic Volume**, originalmente disponibilizada no UCI Machine Learning Repository. O conjunto compreende dados horários do volume de tráfego da rodovia I-94 de 2012 a 2018. Foram trabalhadas variáveis meteorológicas (chuva, neve, temperatura) e descrições climáticas.

### B) Exploração e Pré-processamento
- Foi constatado que a base de dados possuía buracos sistêmicos massivos (*gaps* de até 334 dias), inviabilizando o uso de modelos autorregressivos clássicos (ARIMA/SARIMA).
- Os valores da temperatura absoluta (em Kelvin) com discrepâncias de zero absoluto foram corrigidos por interpolação temporal.
- Dados ausentes numéricos foram preenchidos matematicamente através de Imputação via Mediana (`SimpleImputer`), ajustados exclusivamente no conjunto de Treino para evitar o vazamento de dados (*Data Leakage*).
- Ocorreu a normalização através do `StandardScaler`.

### C) Seleção de Features e Engenharia
- Extraímos e desmembramos as datas em variáveis independentes (hora, dia da semana, mês, ano).
- Aplicamos codificação cíclica (Seno e Cosseno) nas variáveis temporais para que as redes neurais e algoritmos baseados em árvores compreendessem a transição de horários (ex: 23h para 00h).
- Excluímos colunas textuais originais após extrairmos booleanos ricos (ex: `is_raining`, `is_snowing`, `is_rush_hour`).

### D) Modelos de Regressão
Foi construída uma arquitetura de avaliação contendo seis abordagens distintas para cobrir todo o espectro do Aprendizado de Máquina:
1. **Regressão Clássica**: Ridge Regression.
2. **Ensembles (Bagging)**: Random Forest.
3. **Ensembles (Boosting)**: XGBoost, LightGBM e CatBoost.
4. **Redes Neurais**: Multi-Layer Perceptron (MLP).

### E) Implementação e Métricas
- Linguagem Python com as bibliotecas `scikit-learn`, `xgboost`, `lightgbm` e `catboost`.
- Para o Tuning (otimização de hiperparâmetros), utilizou-se o framework Bayesiano **Optuna**.
- A métrica de otimização durante a busca Bayesiana foi controlada pelo particionamento temporal (`TimeSeriesSplit`), garantindo a ausência de *look-ahead bias*.
- **Métricas de Avaliação Utilizadas**: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), $R^2$ (Coeficiente de Determinação), MAPE (Erro Percentual Absoluto Médio) e o Erro Máximo (*Max Error*).

## IV. Experimentos

### IV.1) Experimentos Realizados e Metodologia de Validação
Os dados foram cronologicamente separados em Treino (70%), Validação (15%) e Teste Final (15%). O conjunto de Validação foi utilizado para as buscas profundas de hiperparâmetros. Após selecionar o melhor algoritmo na Validação, o conjunto de Teste foi utilizado em uma avaliação final e definitiva.

O algoritmo vencedor foi o **Random Forest**, sendo então submetido a uma otimização rigorosa de **150 tentativas (trials) no Optuna**, convergindo para 100 estimadores e profundidade máxima de 9.

### IV.2) Resultados
- Na fase de Validação, os algoritmos baseados em árvores (Boosting e Bagging) superaram amplamente as abordagens lineares. O Random Forest obteve um MAPE na casa dos **10,6%**, em contraste drástico com os 51% do modelo Ridge.
- **Teste Definitivo no Holdout (The Vault)**: O modelo consolidado do Random Forest generalizou brilhantemente, apresentando as seguintes métricas no conjunto isolado:
  - **RMSE**: 440,98 carros/hora
  - **$R^2$**: 0,9502
  - **MAPE**: 10,71%
- O $R^2$ indica que a combinação de features temporais cíclicas com algoritmos não-lineares conseguiu captar e explicar **95%** do fluxo caótico e orgânico da rodovia interestadual.
- O Erro Máximo apresentou valores extremos superando a marca de 5.000. Isso evidencia as limitações clássicas em problemas estocásticos de vida real: anomalias massivas (como engavetamentos ou fechamentos de pista) reduzem bruscamente o fluxo, mas o modelo continua a prever o fluxo baseado no clima regular e no horário, visto que não recebe *features* em tempo real de incidentes de trânsito.

## V. Conclusão
O projeto demonstrou que as técnicas de Aprendizado de Máquina moderno (como Random Forest, auxiliado por Otimização Bayesiana profunda via Optuna) são incrivelmente eficazes para séries temporais que possuem padrões sazonais (horários de rush diários). Atingir apenas ~10% de margem de erro usando majoritariamente um termômetro e um relógio ressalta o poder predatório destes algoritmos.
Como trabalhos futuros, propõe-se a injeção de *streams* de dados de sistemas de GPS colaborativo (como o Waze) para anular o Erro Máximo em dias de acidentes imprevisíveis.

## VI. Referências
1. (Artigos e Livros usados ao longo do desenvolvimento).
