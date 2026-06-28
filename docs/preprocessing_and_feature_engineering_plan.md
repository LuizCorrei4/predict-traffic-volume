# Preprocessing and Feature Engineering Plan

Based on the Exploratory Data Analysis (EDA) insights documented in [01-exploration_insights.md](file:///home/gabyl/projetos/predict-traffic-volume/notebooks/01-exploration_insights.md), this plan details the decisions, steps, and structure to build modular preprocessing and feature engineering pipelines in [src](file:///home/gabyl/projetos/predict-traffic-volume/src).

---

## 1. Pipeline Architecture

We will organize the code into two main modules within `src/`:

1.  **`src/data/preprocessing.py`**:
    *   Responsible for cleaning raw data, resolving duplicate timestamps, filtering outliers, dropping the critical missing-data period, and imputing small gaps.
2.  **`src/features/build_features.py`**:
    *   Responsible for temporal engineering (cyclical encodings, weekend/rush-hour flags) and weather feature engineering (binary flags from descriptions, category grouping).

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
    *   *Action*: Reindex the dataframe to a complete hourly frequency index. (Reindexing means forcing the dataset to have a continuous time sequence. If the data jumps from 10:00 directly to 13:00, reindexing will explicitly insert empty rows for "11:00" and "12:00", initially filled with `NaN` values. This transforms "invisible" time gaps into explicitly empty rows).
    *   Once these empty rows are created, impute (fill) gaps of up to 2 consecutive hours using **linear interpolation** for numerical values and **forward fill (`ffill`)** (repeating the last known value) for categorical variables.
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



---

## 4. Methodological Details & Scientific Foundations

### A. Linear Interpolation for Time Series Imputation
Linear interpolation is a mathematical method of estimating values at missing data points by drawing a straight line between the closest surrounding known values. 

**Mathematical Formula & How it Works:**
Let $x_1$ and $x_2$ be the timestamps (represented as numerical hours) of the last known observation before the gap and the first known observation after the gap. Let $y_1$ and $y_2$ be their corresponding known values (e.g., temperature). For a missing timestamp $x$ inside the gap, the estimated value $y$ is computed as:
$$y = y_1 + \frac{x - x_1}{x_2 - x_1} \cdot (y_2 - y_1)$$

In simple terms, the formula calculates the total change in the target variable ($y_2 - y_1$) divided by the total time passed ($x_2 - x_1$) to find the "rate of change per hour". It then applies this rate to figure out the exact value at the missing hour.

**Numerical Example:**
Imagine we are tracking Traffic Volume, and a sensor failed at 14:00.
*   At 13:00 ($x_1$), the traffic was 4000 cars ($y_1$).
*   At 15:00 ($x_2$), the traffic was 5000 cars ($y_2$).
*   We want to estimate the traffic at the missing hour, 14:00 ($x$).

Using the formula:
$$y = 4000 + \frac{14 - 13}{15 - 13} \cdot (5000 - 4000)$$
$$y = 4000 + \frac{1}{2} \cdot (1000) = 4000 + 500 = 4500$$

The algorithm logically draws a straight line between 4000 and 5000, determining that at the exact midpoint (14:00), the traffic was exactly 4500 cars.

**Conceptual Justification:**
*   **Smooth Transitions**: Physical and environmental parameters (like temperature, cloud cover, and general weather transitions) do not change instantaneously; they fluctuate smoothly. Assuming a constant rate of change between adjacent hours is a highly robust approximation.
*   **Threshold-Based Limits ($\le 2$ hours)**: While linear interpolation is simple and effective, its accuracy diminishes as gap size increases because it cannot capture non-linear daily patterns or sudden weather shifts over longer durations (Moritz et al., 2017). By strictly limiting imputation to short gaps ($\le 2$ hours) and dropping larger blocks, we preserve dataset integrity and prevent the injection of synthetic bias.

### B. Cyclical Encoding of Temporal Features (Trigonometric Encodings)
When we use plain numbers to represent time, like $0$ for Midnight and $23$ for 11 PM, machine learning models get confused. To a model, the mathematical distance between $0$ and $23$ is very large (23 units). However, in the real world, 11 PM and Midnight are just 1 hour apart! This creates an "artificial jump" where the end of the day doesn't connect smoothly back to the start of the next day.

**The Solution: The Clock Metaphor**
To fix this, we map the hours onto a circle, exactly like the face of a clock. Instead of using a single number from 0 to 23, each hour becomes a 2D coordinate $(x, y)$ on that circular clock. 
- The $x$-coordinate is calculated using the **cosine** of the angle.
- The $y$-coordinate is calculated using the **sine** of the angle.

**Mathematical Formula:**
For a time variable $x$ with a known cycle period $T$ (where $T=24$ for hours, and $T=7$ for days of the week), we transform it into:
$$x_{\text{sin}} = \sin\left(\frac{2\pi \cdot x}{T}\right)$$
$$x_{\text{cos}} = \cos\left(\frac{2\pi \cdot x}{T}\right)$$

**Why is this highly beneficial?**
*   **Perfect Continuity**: On our mathematical "clock face", the coordinate point for 23:00 sits right next to the coordinate point for 00:00. The model now correctly understands that 11 PM and Midnight are physically close to each other, completely eliminating the artificial jump.
*   **Resolving Ambiguities (Why we need both Sin and Cos)**: If we only use the *sine* (the vertical height on the clock), the model can't tell the difference between times that share the same height (like 3 PM and 9 PM on a 24h cycle). We must provide *both* sine and cosine so the model has the exact unique 2D position on the circle.
*   **Better Predictions**: Traffic volume is deeply cyclical (repeating every 24 hours and every 7 days). By feeding the model a continuous geometric loop instead of a broken number line, it can learn the daily and weekly repeating patterns much more smoothly and accurately.

---

## 5. Execution Summary

The pipeline described in this plan has been successfully implemented and tested:

1.  **Preprocessing Script**: [`src/data/preprocessing.py`](file:///home/gabyl/projetos/predict-traffic-volume/src/data/preprocessing.py) implements outlier removal (setting invalid $0\text{ K}$ temperatures and $>500\text{ mm}$ rain values to `NaN`), reindexing to a continuous hourly timescale, and applying linear/ffill interpolation to gaps $\le 2$ hours.
2.  **Feature Script**: [`src/features/build_features.py`](file:///home/gabyl/projetos/predict-traffic-volume/src/features/build_features.py) implements the cyclical encoding transforms, weekend/rush hour flags, and binary weather indicators.
3.  **Pipeline Orchestrator**: The CLI tool [`src/run_pipeline.py`](file:///home/gabyl/projetos/predict-traffic-volume/src/run_pipeline.py) runs the end-to-end flow (using `poetry run task preprocess`), drops target-missing rows, and saves the cleaned dataset to `data/processed/clean_traffic_data.csv` (`42,880` rows, `23` features).

---

## 6. References

*   **Moritz, S., Sardá, A., Bartz-Beielstein, T., Griensven, M., & Zaefferer, M. (2017).** *Comparison of different methods for univariate time series imputation in R.* bioRxiv. (Underpins the decision to use linear interpolation for short gaps and discard larger gaps to avoid model bias).
*   **NVIDIA Technical Blog.** *Three Approaches to Encoding Time Information as Features for Machine Learning Models.* (Validates trigonometric sine-cosine transformations as a best-practice baseline for circular time representations).
