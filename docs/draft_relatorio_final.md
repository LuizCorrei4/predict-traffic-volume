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
Os dados foram cronologicamente separados em Treino (70%), Validação (15%) e Teste Final (15%). O conjunto de Validação foi utilizado para a busca profunda de hiperparâmetros no Optuna. O conjunto de Teste foi mantido estritamente isolado para validação do campeão.

**(Incluir Tabelas 1 e 2 contendo Hiperparâmetros otimizados e Performance de cada classificador - Os gráficos em barra a serem construídos entrarão aqui)**

### IV.2) Resultados
*(Espaço reservado para a discussão aprofundada dos resultados, justificando por que algoritmos baseados em Boosting como XGBoost e LightGBM perfomaram com MAPE na casa de 10% a 15%, superando os modelos lineares que chegaram a 50%. Comentar também sobre os picos e anomalias capturados no Max_Error.)*

## V. Conclusão
*(Síntese técnica e de negócio. Destacar que prever o tráfego de milhares de seres humanos tem alta estocasticidade e atingir os erros encontrados é notável. Falar das limitações do dataset, como acidentes que não estão mapeados nas features e propor como trabalho futuro a integração com APIs de GPS em tempo real).*

## VI. Referências
1. (Artigos e Livros usados ao longo do desenvolvimento).
