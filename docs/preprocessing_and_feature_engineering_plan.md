# Preprocessing and Feature Engineering Plan

Based on the Exploratory Data Analysis (EDA) insights documented in [01-exploration_insights.md](file:///home/gabyl/projetos/predict-traffic-volume/notebooks/01-exploration_insights.md), this plan details the decisions, steps, and structure to build modular preprocessing and feature engineering pipelines in [src](file:///home/gabyl/projetos/predict-traffic-volume/src).

---

## 1. Pipeline Architecture

We will organize the code into two main modules within `src/`:

1.  **`src/data/preprocessing.py`**:
    *   Responsible for cleaning raw data, resolving duplicate timestamps, filtering outliers, dropping the critical missing-data period, and imputing small gaps.
2.  **`src/features/build_features.py`**:
    *   Responsible for temporal engineering (cyclical encodings, weekend/rush-hour flags), weather feature engineering (binary flags from descriptions, category grouping), feature scaling (Standardization/Normalization), and dataset splitting.

---

## 2. Preprocessing Strategy (`src/data/preprocessing.py`)

### A. Duplicate Timestamp Resolution
*   **Rule**: Timestamps are unique hourly slots. Group duplicate rows by `date_time` (index).
*   **Action**: 
    *   Apply `mean()` to numeric columns: `temp`, `rain_1h`, `snow_1h`, `clouds_all`, and `traffic_volume`.
    *   Apply `first()` to categorical columns: `holiday`, `weather_main`, and `weather_description`.

### B. Outlier Treatment
*   **Temperature (`temp`)**:
    *   *Issue*: 10 instances of `0 K` (absolute zero), which are physically impossible.
    *   *Action*: Replace values $< 100\text{ K}$ with `NaN`, then interpolate using surrounding hours.
*   **Rainfall (`rain_1h`)**:
    *   *Issue*: 1 instance of `9831.3 mm`, which is a clear sensor anomaly.
    *   *Action*: Replace values $> 500\text{ mm}$ with `NaN`, then interpolate.

### C. Missing Data & Gap Resolution
*   **Critical Gap (August 2014 – June 2015)**:
    *   *Decision*: **Drop this entire period** from the dataset. The gap is too large to impute without introducing high bias, and it breaks temporal models.
*   **Small Gaps ($\le 2$ hours)**:
    *   *Action*: Reindex the dataframe to a complete hourly frequency index. Impute gaps of 1 or 2 consecutive hours using linear interpolation for numerical values, and forward fill (`ffill`) for categorical values.
*   **Large Gaps ($> 2$ hours)**:
    *   *Action*: Retain them as `NaN` (or mask them) during feature creation. Drop these rows prior to model training since we cannot train without a target (`traffic_volume`) or valid lag features.

---

## 3. Feature Engineering Strategy (`src/features/build_features.py`)

### A. Temporal Features
*   **Datetime Parsing**: Extract `hour`, `day_of_week`, `month`, and `year` from the timestamp.
*   **Cyclical Encodings**:
    *   Traffic is highly cyclical. We will compute sine and cosine transformations of cyclical features:
        $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
        $$\text{day\_sin} = \sin\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right), \quad \text{day\_cos} = \cos\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right)$$
*   **Weekend Flag**: Create `is_weekend` (1 if Saturday or Sunday, else 0).
*   **Rush Hour Flag**: Create `is_rush_hour` (1 if weekday between 7-9 AM or 3-6 PM, else 0).

### B. Categorical Weather Encodings
*   **Holiday**: Convert the high-sparsity `holiday` text into a binary `is_holiday` flag (1 if a holiday name is present, else 0).
*   **Weather Indicators (Hierarchical Encodings)**:
    *   *Decision*: Use `weather_main` as the primary categorical encoder (One-Hot Encoded).
    *   Derive robust binary flags from `weather_description` to handle quality issues in numeric columns:
        *   `is_raining` (1 if `rain_1h > 0` or description contains "rain" or "drizzle", else 0).
        *   `is_snowing` (1 if `snow_1h > 0` or description contains "snow" or "sleet", else 0) — *this resolves the 2,264 false-zero snow sensor errors*.
        *   `is_foggy_misty` (1 if description contains "fog", "mist", or "haze", else 0).

### C. Scaling and Transformations (Deferred)
**Important Note:** To prevent **Data Leakage**, scalers and encoders must never be fitted on the entire dataset. Therefore, the following steps are defined as functions but are **not** executed during the automated feature engineering pipeline. They will be applied only *after* splitting the data into training, validation, and testing sets.
*   **Numerical Features**:
    *   `temp`: Standardize using `StandardScaler` (Z-score normalization).
    *   `clouds_all`: Min-Max scale to $[0, 1]$ or keep raw.
    *   `rain_1h` (optional raw): Apply log transform $\log(1 + \text{rain\_1h})$ due to its extreme right-skewness.
*   **Categorical Features**:
    *   Apply `OneHotEncoder` to `weather_main`.

---

## 4. Methodological Details & Scientific Foundations

### A. Linear Interpolation for Time Series Imputation
Linear interpolation is a mathematical method of estimating values at missing data points by drawing a straight line between the closest surrounding known values. 

**Mathematical Formula:**
Let $x_1$ and $x_2$ be the timestamps of the last known observation before the gap and the first known observation after the gap, respectively, with $y_1$ and $y_2$ being their corresponding known values (e.g., temperature). For any missing timestamp $x$ in the gap ($x_1 < x < x_2$), the estimated value $y$ is computed as:
$$y = y_1 + \frac{x - x_1}{x_2 - x_1} \cdot (y_2 - y_1)$$

**Conceptual Justification:**
*   **Smooth Transitions**: Physical and environmental parameters (like temperature, cloud cover, and general weather transitions) do not change instantaneously; they fluctuate smoothly. Assuming a constant rate of change between adjacent hours is a highly robust approximation.
*   **Threshold-Based Limits ($\le 2$ hours)**: While linear interpolation is simple and effective, its accuracy diminishes as gap size increases because it cannot capture non-linear daily patterns or sudden weather shifts over longer durations (Moritz et al., 2017). By strictly limiting imputation to short gaps ($\le 2$ hours) and dropping larger blocks, we preserve dataset integrity and prevent the injection of synthetic bias.

### B. Cyclical Encoding of Temporal Features (Trigonometric Encodings)
Representing cyclical data (e.g., hours of the day or days of the week) as simple numerical sequences ($0, 1, 2, ..., 23$) introduces artificial discontinuities. Under a naive integer encoding, the distance between $23$ (11 PM) and $0$ (Midnight) is $23$ units, whereas in reality, they are only $1$ hour apart.

**Mathematical Formula:**
For a time variable $x$ with a known cycle period $T$ (where $T=24$ for hours, and $T=7$ for days of the week), we transform $x$ into two features representing $x$ and $y$ coordinates on a unit circle:
$$x_{\text{sin}} = \sin\left(\frac{2\pi \cdot x}{T}\right)$$
$$x_{\text{cos}} = \cos\left(\frac{2\pi \cdot x}{T}\right)$$

**Conceptual Justification:**
*   **Preserving Circular Continuity**: By projecting the 1D time feature into 2D coordinates on a unit circle, we align the end of the cycle (Midnight) perfectly next to the beginning of the next cycle. The Euclidean distance between $23:00$ and $00:00$ correctly contracts to the same distance as between $00:00$ and $01:00$.
*   **Model Compatibility**: Models (especially Neural Networks and linear architectures) can interpret this coordinate space as a continuous cyclical continuum, preventing the model from assuming midnight is "mathematically distant" from 11 PM.
*   **Use of Pairs**: Both sine and cosine components must be used together. Using only one component (e.g., sine) creates ambiguity, as two different times in a cycle can yield the same sine value (e.g., $\sin(\theta)$ at $30^\circ$ and $150^\circ$ are identical). The cosine resolves this ambiguity by providing the orientation.

---

## 5. Execution Summary

The pipeline described in this plan has been successfully implemented and tested:

1.  **Preprocessing Script**: [`src/data/preprocessing.py`](file:///home/gabyl/projetos/predict-traffic-volume/src/data/preprocessing.py) implements outlier removal (setting invalid $0\text{ K}$ temperatures and $>500\text{ mm}$ rain values to `NaN`), reindexing to a continuous hourly timescale, and applying linear/ffill interpolation to gaps $\le 2$ hours.
2.  **Feature Script**: [`src/features/build_features.py`](file:///home/gabyl/projetos/predict-traffic-volume/src/features/build_features.py) implements the cyclical encoding transforms, weekend/rush hour flags, and binary weather indicators. It also contains scaling functions that are kept isolated for future model training stages.
3.  **Pipeline Orchestrator**: The CLI tool [`src/run_pipeline.py`](file:///home/gabyl/projetos/predict-traffic-volume/src/run_pipeline.py) runs the end-to-end flow (using `poetry run task preprocess`), drops target-missing rows, and saves the cleaned dataset to `data/processed/clean_traffic_data.csv` (`42,880` rows, `23` features).

---

## 6. References

*   **Moritz, S., Sardá, A., Bartz-Beielstein, T., Griensven, M., & Zaefferer, M. (2017).** *Comparison of different methods for univariate time series imputation in R.* bioRxiv. (Underpins the decision to use linear interpolation for short gaps and discard larger gaps to avoid model bias).
*   **NVIDIA Technical Blog.** *Three Approaches to Encoding Time Information as Features for Machine Learning Models.* (Validates trigonometric sine-cosine transformations as a best-practice baseline for circular time representations).
*   **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning.* Springer. (Supports standard scaling normalization and one-hot encoding choices to avoid numerical dominance in regression models).
