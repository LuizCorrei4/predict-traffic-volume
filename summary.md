# Project Summary: Predict Traffic Volume (USP Machine Learning)

**Context:** This file serves as the context transfer for a new conversation session. It summarizes the entire development, methodology, and final state of the Machine Learning project for the SCC-276 (Aprendizado de Máquina) course at USP.

---

## 1. Project Overview
- **Goal:** Predict the continuous `traffic_volume` of the I-94 Interstate highway (Minnesota) using meteorological (weather, temperature, rain, snow) and temporal (hour, day, month, holiday) features.
- **Dataset:** *Metro Interstate Traffic Volume* (Kaggle/UCI). It contains 48,204 hourly records from 2012 to 2018.
- **Challenge:** The dataset contains massive temporal gaps (e.g., 334 days missing), making classical autoregressive models (ARIMA) impossible. Traffic is highly stochastic due to human behavior and external shocks (accidents).

## 2. Methodology & Technical Decisions
- **Anti-Data Leakage Pipeline:** The data is strictly partitioned chronologically (70% Train, 15% Validation, 15% Test/"The Vault"). Imputers (`SimpleImputer` with median) and Scalers (`StandardScaler`) are strictly fitted on the **Train set only**.
- **Feature Engineering (Cyclical Encoding):** To help ML models understand that 23:00 is next to 00:00, temporal features (hour, day of week, month) were converted into Trigonometric cyclical features (`_sin` and `_cos`).
- **Models Implemented:** Ridge Regression (baseline), Random Forest, XGBoost, LightGBM, CatBoost, and Multi-Layer Perceptron (MLP).
- **Hyperparameter Tuning:** We built an orchestrator (`src/models/train_model.py`) that uses **Optuna** to run Bayesian searches (with `TimeSeriesSplit` cross-validation to prevent look-ahead bias). 

## 3. Final Results ("The Vault")
The top performing algorithm was the **Random Forest Regressor**. It was optimized using 150 trials in Optuna on the Validation set. Upon final evaluation on the isolated Test Set, it achieved state-of-the-art results:
- **$R^2$:** 0.9502 (It explains 95% of the highway traffic variance)
- **MAPE:** 10.71% 
- **RMSE:** 440.98 cars/hour
- All visual reports (Metrics Comparison, Full Test Series, 30-Day, 7-Day, and 3-Day Micro-trends) are saved in `reports/figures/`.

## 4. Current State of the Repository
The repository is perfectly structured, fully modular, committed, and ready for academic submission:
- `src/`: Contains production-grade object-oriented code (`run_pipeline.py`, `models/data_splitter.py`, `models/model_factory.py`, `models/evaluator.py`, `models/train_model.py`, `models/plot_metrics.py`).
- `contexto_trabalho_AM_USP/`: Contains the USP requirements and the final academic report (`draft_relatorio_final.md`) written at a deep scientific level, including references to Uber (Laptev et al.), Optuna (Akiba et al.), NVIDIA (cyclical encoding), and Moritz et al. (imputation). Also includes the presentation slides structure.
- `entrega_am_15639682_LuizCorreia/`: An isolated delivery folder prepared specifically for the Professor/Monitor to evaluate the code, containing a custom `LEIA-ME_primeiro.md` map.

## 5. Next Steps for the User
1. **Report Delivery:** Copy the contents of `contexto_trabalho_AM_USP/draft_relatorio_final.md` into an Overleaf `.tex` template.
2. **Slides/Presentation:** Use `contexto_trabalho_AM_USP/slides_apresentacao.md` to build a 10-minute presentation emphasizing the lack of data leakage, the cyclical trigonometric features, and the high $R^2$ on the Random Forest.
3. **e-Disciplinas:** Submit the `.zip` from the `entrega_am_15639682_LuizCorreia/` folder.
