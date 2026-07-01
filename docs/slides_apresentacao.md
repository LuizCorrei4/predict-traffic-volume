# Apresentação do Projeto: Previsão de Volume de Tráfego

*(Este documento serve como roteiro para a montagem dos slides da apresentação final da disciplina SCC-276)*

---

## Slide 1: Capa
- **Título**: Previsão de Volume de Tráfego na Interestadual I-94: Uma Abordagem com Machine Learning
- **Disciplina**: SCC-276 - Aprendizado de Máquina (USP)
- **Professor(a)**: Roseli Aparecida Francelin Romero
- **Membros**: [Seus Nomes]

---

## Slide 2: O Desafio (Introdução)
- **Contexto**: O crescimento populacional e logístico torna as rodovias gargalos vitais. Prever o tráfego permite otimizar rotas, mitigar congestionamentos e reduzir acidentes.
- **O Problema Analisado**: Estimar a quantidade de carros (Volume de Tráfego) por hora na rodovia I-94 (Minnesota, EUA) utilizando apenas o histórico de horário e clima.
- **Dificuldade**: Comportamento humano é estocástico (imprevisível em nível micro).

---

## Slide 3: A Base de Dados
- **Dataset**: *Metro Interstate Traffic Volume* (Kaggle / UCI Machine Learning).
- **Tamanho**: Mais de 48.000 registros (de 2012 a 2018).
- **Features (Atributos)**: 
  - *Climáticas*: Temperatura (Kelvin), Chuva (mm), Neve (mm), % Nuvens.
  - *Temporais*: Data e Hora (`date_time`).
  - *Eventos*: Feriados (Nacionais e Regionais).
- **Target**: `traffic_volume` (Numérico).

---

## Slide 4: O Abismo Temporal (EDA)
- **Descoberta**: Analisando os dados, encontramos buracos massivos (ex: 334 dias seguidos sem medições de tráfego entre 2014 e 2015).
- **Conclusão**: Impossível utilizar modelos tradicionais de séries temporais como ARIMA. Solução adotada: transformar a predição temporal pura em um problema robusto de Regressão usando Machine Learning Clássico e *Ensembles*.

---

## Slide 5: Engenharia Matemática (Features)
- **Problema de Vazamento (Leakage)**: O conjunto foi dividido estritamente no tempo (Treino 70% | Validação 15% | Teste 15%). Escalamento (`StandardScaler`) aplicado só no treino.
- **Codificação Cíclica**: 
  - Como a máquina sabe que a hora 23:00 fica encostada na hora 00:00?
  - Solução: Criamos variáveis de Seno e Cosseno para a hora do dia e meses do ano, transformando o relógio em um círculo matemático que os modelos de IA entendem perfeitamente.

---

## Slide 6: Modelos Implementados
Exploramos a fundo o espectro do Machine Learning com **6 Algoritmos** implementados do zero:
1. **Regressão Clássica**: Regressão Ridge
2. **Rede Neural**: Multi-Layer Perceptron (MLP)
3. **Bagging**: Random Forest
4. **Boosting de Alta Eficiência**: XGBoost, LightGBM, CatBoost

*(Todos orquestrados usando Otimização Bayesiana - **Optuna**)*

---

## Slide 7: Desempenho Geral (Os Campeões)
*(Inserir aqui a imagem `reports/figures/metrics_comparison.png`)*
- **Resultados de Validação**: 
  - Ridge (Linear) não suportou a complexidade: MAPE de 51%
  - Família Árvores de Decisão dominou!
  - **Random Forest**: O Grande Vencedor.
- A métrica `R²` do Random Forest no teste final atingiu incríveis **0.9502** (explicou 95% do comportamento da rodovia).

---

## Slide 8: Desvendando o "The Vault" (Test Set)
Ao expor o Random Forest otimizado aos dados isolados de teste (o que ele nunca tinha visto):
- **MAPE**: ~10.7% (Erro de 10% em comportamento humano coletivo é estado-da-arte).
- **RMSE**: ~440 carros/hora (de uma média de ~3.200).
- **Erro Máximo (Anomalias)**: Chegou a 5.000. Por quê? Porque acidentes de trânsito ou pistas bloqueadas *não constam no dataset de clima*, derrubando o volume real para zero e gerando picos isolados de erro.

---

## Slide 9: Visualizando a Assertividade (1 Mês e 1 Semana)
*(Inserir as imagens `reports/figures/RANDOM_FOREST_ts_month.png` e `_ts_week.png`)*
- O modelo consegue capturar fielmente os picos da manhã e da tarde (Horas do Rush).
- Consegue deduzir a queda brutal de fluxo aos finais de semana e madrugadas.

---

## Slide 10: O Pulo do Gato (Microtendências - 3 Dias)
*(Inserir a imagem `reports/figures/RANDOM_FOREST_ts_3days.png`)*
- O recorte de 3 dias prova que as Árvores de Decisão (com as horas convertidas em senos e cossenos) rastreiam milimetricamente o comportamento de subida e descida diária, ajustando-se à chuva e variação de temperatura.

---

## Slide 11: Conclusão
- Machine Learning (com Boosting e Bagging) é formidável para problemas temporais desde que a Engenharia de Features transforme a dimensão tempo matematicamente.
- Prever milhares de carros com ~10% de erro usando apenas um termômetro e um relógio é um feito notável.
- **Trabalhos Futuros**: Integrar dados em tempo real da polícia rodoviária para reduzir o erro máximo em dias de acidentes severos.

---

## Slide 12: Muito Obrigado!
- **Dúvidas?**
- Repositório completo com código modular, testes unitários, orquestrador e reprodutibilidade científica.
