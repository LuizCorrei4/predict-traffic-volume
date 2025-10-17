# predict-traffic-volume
Predicting traffic volume on the I-94 Interstate highway.

## Dataset

This project uses the **Metro Interstate Traffic Volume** dataset from the UCI Machine Learning Repository. This dataset contains hourly westbound traffic volume data for the I-94 Interstate highway, recorded at the MN DoT ATR station 301 (located roughly midway between Minneapolis and St. Paul, Minnesota).

The data was collected from October 2012 to January 2018 and includes weather and holiday information.

- **Original Source:** [UCI Machine Learning Repository: Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic-volume)
- **Instances:** 48,204
- **Task:** Regression (predicting `traffic_volume`)

### Attributes

The dataset includes the following columns:

| Atributo | Descrição | Tipo |
| :--- | :--- | :--- |
| `holiday` | US National holidays and regional holidays (e.g., Minnesota State Fair) | Categorical |
| `temp` | Average temperature in Kelvin | Numeric |
| `rain_1h` | Amount of rain in mm that occurred in the hour | Numeric |
| `snow_1h` | Amount of snow in mm that occurred in the hour | Numeric |
| `clouds_all` | Percentage of cloud cover | Numeric |
| `weather_main` | Short textual description of the current weather | Categorical |
| `weather_description` | Longer textual description of the current weather | Categorical |
| `date_time` | Hour of the data collected (in local CST time) | DateTime |
| **`traffic_volume`** | **(Target)** Hourly I-94 ATR 301 reported westbound traffic volume | Numeric |




