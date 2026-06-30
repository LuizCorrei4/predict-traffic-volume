# Project References & Scientific Foundations

This document catalogs the academic papers, articles, and technical blogs that provide the theoretical foundation and methodological justification for the decisions made throughout the `predict-traffic-volume` project.

## 1. Time Series Imputation Strategies

*   **Title:** Comparison of different methods for univariate time series imputation in R
*   **Authors:** Steffen Moritz, Alexis Sardá, Thomas Bartz-Beielstein, Martin Zaefferer, Jörg Stork
*   **Link:** [https://arxiv.org/abs/1510.03924](https://arxiv.org/abs/1510.03924) or [bioRxiv](https://www.biorxiv.org/content/10.1101/059311v1)
*   **Concept & Project Application:** 
    This paper comprehensively compares imputation methods (like linear interpolation, spline, and kalman smoothing) on time series with varying gap sizes. The study demonstrates that while complex methods might perform well on structural missing data, simple linear interpolation is highly robust for very small gaps, whereas *any* imputation method degrades rapidly and introduces severe bias on large gaps (especially in highly seasonal data). 
    *   **Project Decision:** We adopted a strict limit of $\le 2$ hours for linear interpolation of our traffic data. Any gap larger than 2 hours is strictly left as missing and subsequently dropped. This prevents the injection of synthetic, artificially flat data into our training set, preserving the integrity of the daily rush-hour cycles.

## 2. Temporal Feature Engineering (Cyclical Encodings)

*   **Title:** Three Approaches to Encoding Time Information as Features for Machine Learning Models
*   **Author:** NVIDIA Technical Blog
*   **Link:** [https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-machine-learning-models/](https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-machine-learning-models/)
*   **Concept & Project Application:**
    Standard machine learning algorithms interpret numerical values linearly. If we pass the hour of the day as integers from 0 to 23, the model perceives a mathematical jump of 23 units between 23:00 (11 PM) and 00:00 (Midnight), even though they are only 1 hour apart physically. This article details the mathematical projection of temporal features onto a unit circle using Sine and Cosine transformations.
    *   **Project Decision:** We explicitly mapped the `hour` and `day_of_week` variables into `hour_sin`, `hour_cos`, `day_sin`, and `day_cos`. This continuous geometric representation perfectly aligns with the cyclical nature of traffic volume, allowing models like XGBoost and Neural Networks to seamlessly learn daily and weekly patterns without discontinuities.

## 3. Forecasting Architecture: Machine Learning vs. Classical Time Series

*   **Title:** Forecasting at Uber: An Introduction
*   **Authors:** Nikolay Laptev, Slawek Smyl, Santhosh Shanmugam
*   **Link:** [https://www.engr.psu.edu/xinli/IFEE511/laptev17.pdf](https://www.engr.psu.edu/xinli/IFEE511/laptev17.pdf) (or Uber Engineering Blog)
*   **Concept & Project Application:**
    Classical time series models like ARIMA or SARIMA were designed for low-frequency data (monthly, quarterly) with a single seasonality. Traffic volume is high-frequency (hourly) data exhibiting multiple seasonalities (daily and weekly) and is heavily influenced by exogenous variables (weather, holidays). Uber's forecasting team outlines that for complex, high-frequency, multivariate problems, Machine Learning (Tree-based models and Neural Networks) vastly outperforms traditional statistical models because they can ingest exogenous features and handle complex nonlinear interactions effortlessly without requiring strict continuity.
    *   **Project Decision:** Because our dataset is punctuated by 182 structural time jumps (gaps $>1$ hour) and heavily relies on boolean weather flags, we abandoned ARIMA/SARIMA. Instead, we architected our modeling pipeline to rely on Machine Learning algorithms (Linear Regression, XGBoost, Multi-Layer Perceptrons) combined with a Chronological Holdout split.
