# Próximos Passos (Para Amanhã) 🚀

Este documento centraliza as últimas ações manuais e burocráticas que você deve fazer para entregar o trabalho da USP (SCC-276) com maestria.

## 1. Entrega no e-Disciplinas
Você precisa compactar parte deste repositório para o sistema da USP.
A Profa. Roseli e os monitores precisam conseguir ler e rodar o projeto de maneira fácil.
**O que fazer:**
- Crie uma pasta temporária (ex: `entrega_am_usp_seu_nome/`).
- Copie para dentro dela:
  - A pasta `src/` (onde toda a magia acontece de verdade).
  - A pasta `reports/figures/` (provando que extraímos gráficos de ponta).
  - O `pyproject.toml` e `poetry.lock` (para reprodutibilidade).
  - O arquivo `README.md` raiz (ele já está excelente e em inglês explicando tudo).
  - **Dica de Ouro:** Adicione um pequeno README complementar em português (`LEIA-ME_USP.md`) instruindo o monitor a olhar o `README.md` principal, e explicando brevemente como você construiu o pipeline de avaliação (`train_model.py` com `Optuna`).
- Compacte a pasta gerando um `.zip`.

## 2. Redação do Artigo no Overleaf (.tex)
O relatório final acadêmico oficial em PDF (aquele que pesa na nota).
**O que fazer:**
- Abra o Overleaf e crie um projeto com o template da SBC (ou o sugerido pela disciplina).
- Abra o nosso arquivo `docs/draft_relatorio_final.md`. Todo o roteiro, da Introdução à Conclusão, está lá.
- Você precisa literalmente apenas transferir os parágrafos do nosso Markdown para o `.tex`. 
- Na seção **Experimentos (IV.1 e IV.2)**, utilize o comando `\includegraphics{}` do LaTeX para colar as imagens que geramos hoje e que estão salvas em `reports/figures/` (o painel comparativo `metrics_comparison.png` e os recortes temporais de Teste `RANDOM_FOREST_ts_month.png` e `_3days.png`).

## 3. Preparação dos Slides e Apresentação (10 Minutos)
A coroação do seu trabalho. Uma boa apresentação disfarça qualquer falha e eleva projetos ótimos para o nível da excelência.
**O que fazer:**
- Abra o PowerPoint ou Google Slides.
- Leia o arquivo `docs/slides_apresentacao.md`. Eu desenhei exatamente 12 slides com os *bullet points* cruciais.
- Copie e cole os textos. Adicione as imagens correspondentes.
- **Dica de Apresentação:** Em apresentações curtas (10 min), o maior erro é se alongar muito explicando o que é Random Forest (a banca já sabe). **Gaste o seu tempo** provando os seus diferenciais competitivos:
  1. Mostre como você superou a falha no dataset (os buracos de 300 dias) escapando do ARIMA.
  2. Prove metodologicamente que você isolou o conjunto de teste temporalmente (*zero data leakage*).
  3. Mostre o gráfico de recortes de **3 Dias** (as microtendências), destacando como o uso trigonométrico (seno e cosseno para as horas do dia) possibilitou a Árvore de Decisão surfar nas ondas do horário de pico humano de forma brilhante, atingindo uma precisão massiva de **95% (R²)**.

Boa sorte! Vá lá e garanta o 10.