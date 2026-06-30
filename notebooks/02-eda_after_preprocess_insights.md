# EDA Post-Preprocessing: Insights & Feature Selection

After generating the engineered features (cyclical time coordinates, weekend/rush-hour flags, and boolean weather indicators), we ran a second Exploratory Data Analysis to validate these features and make final decisions for the Machine Learning modeling phase.

---

## 1. Validating Temporal Engineering

Our temporal feature engineering was highly successful in capturing the variance of `traffic_volume`.

### A. Rush Hour & Weekend Flags
*   **Rush Hour (`is_rush_hour`)**: The average traffic volume during non-rush hours is **2,719 cars**, while during rush hours it explodes to **5,507 cars**. This flag has a strong positive correlation (`0.57`) with the target.
*   **Weekend (`is_weekend`)**: Weekdays average **3,568 cars**, while weekends drop significantly to **2,627 cars**. This flag has a strong negative correlation (`-0.21`).

### B. Cyclical Encodings (Sine / Cosine)
*   The `hour_cos` feature exhibited a massive correlation of **-0.76** with traffic volume. By transforming the linear `hour` into coordinates on a circle, we provided a mathematical representation that perfectly aligns with the daily ebb and flow of traffic. 

**Decision:** Since we have successfully mapped time to `hour_cos`, `hour_sin`, `day_cos`, `day_sin`, `is_weekend`, and `is_rush_hour`, keeping the raw integer columns (`hour`, `day_of_week`, `month`, `year`) would confuse linear models and add unnecessary dimensionality. **We will drop the raw integer date columns.**

---

## 2. Weather Variables: Redundancy & Multicollinearity

We needed to decide whether to One-Hot Encode the original text column (`weather_main`) or rely solely on our custom boolean flags (`is_raining`, `is_snowing`, `is_foggy_misty`).

### Analyzing Traffic by Weather Conditions:
*   **Snow**: When `weather_main` is "Snow", average traffic is **3,045**. When our custom `is_snowing` flag is 1, average traffic is **3,048**.
*   **Fog/Mist**: When `weather_main` is "Fog" or "Mist", traffic ranges from **2,743 to 2,893**. When our custom `is_foggy_misty` flag is 1, average traffic is **3,024**.
*   **Rain**: Both the main category and our custom flag show traffic hovering around **3,300**.

**Conclusion:**
Our three simple boolean flags successfully capture the exact same statistical shifts in traffic volume as the 11 different categories inside `weather_main`. 

If we keep `weather_main`, we would have to create 11 new One-Hot Encoded columns, massively increasing the dimensionality of our dataset. Worse, because `is_snowing == 1` overlaps almost perfectly with `weather_main == Snow`, we would introduce severe **Multicollinearity**, confusing the machine learning algorithm.

**Decision:** **We will drop both `weather_main` and `weather_description`.** We will rely purely on the boolean flags to keep the model parsimonious (simple and robust).

---

## 3. Final Dataset Format (Sealed)

Based on the EDA and data quality checks, we have reached a consensus on the final schema for the dataset before modeling. We prioritize robust engineered features over sparse or faulty raw data to prevent multicollinearity and confusion in algorithms.

### A. Columns to KEEP
*   **Target:** `traffic_volume`
*   **Continuous/Numeric Features (will be scaled):**
    *   `temp`: Corrected temperature.
    *   `clouds_all`: Cloud cover percentage.
    *   `rain_1h_log`: Log-transformed rainfall (to handle extreme skewness).
*   **Cyclical Temporal Features (Sine/Cosine):**
    *   `hour_sin`, `hour_cos`, `day_sin`, `day_cos`.
*   **Binary Flags (no scaling required):**
    *   `is_weekend`, `is_rush_hour`, `is_holiday`.
    *   `is_raining`, `is_snowing`, `is_foggy_misty`.

### B. Columns to DROP
*   **Raw Dates:** `hour`, `day_of_week`, `month`, `year` (Replaced by cyclical coordinates to avoid non-linear sequence confusion).
*   **Categorical Text:** `weather_main`, `weather_description`, `holiday` (Replaced by robust binary flags. Keeping them would require One-Hot Encoding, introducing massive dimensionality and severe multicollinearity).
*   **Faulty/Skewed Numerics:** 
    *   `snow_1h`: Dropped due to 2,264 "silent failure" sensor errors (where the text description reported snow but the numerical sensor reported 0.0). The engineered `is_snowing` flag is significantly more reliable.
    *   `rain_1h`: Dropped in favor of `rain_1h_log`.

### Next Steps:
We will proceed to split the data (Train/Validation/Test) strictly respecting the temporal order to prevent Data Leakage, and then fit our Scalers purely on the training set.
