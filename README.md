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

## Project Structure

This project follows a standard Data Science project structure:

- `data/`: Contains the raw and processed datasets (e.g., `data/processed/clean_traffic_data.csv`).
- `notebooks/`: Jupyter notebooks:
  - `01-exploration.ipynb`: Exploratory Data Analysis (EDA).
  - `02-preprocessing_and_features.ipynb`: Execution and validation of preprocessing and feature engineering.
- `src/`: Source code of the project containing preprocessing, features, and model modules. **See [src/README.md](file:///home/gabyl/projetos/predict-traffic-volume/src/README.md) for execution commands.**
- `tests/`: Unit tests for python modules.
- `models/`: Stores pre-fitted scalers, encoders, and trained models (e.g., `models/scalers_and_encoders.pkl`).
- `docs/`: Holds project documentation, including the [preprocessing and feature engineering plan](file:///home/gabyl/projetos/predict-traffic-volume/docs/preprocessing_and_feature_engineering_plan.md) with scientific and mathematical justifications.

## How to Clone and Run Locally

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

### Prerequisites

1. **Python 3.12+** must be installed on your system.
2. **Poetry** must be installed. If you don't have it, you can install it by running:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   *(Or refer to the [official Poetry documentation](https://python-poetry.org/docs/#installation) for other installation methods).*

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LuizCorrei4/predict-traffic-volume.git
   cd predict-traffic-volume
   ```

2. **Install dependencies:**
   Run the following command to create a virtual environment and install all the project dependencies (including development ones):
   ```bash
   poetry install
   ```

3. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```

4. **Run the Jupyter Notebooks (Optional):**
   If you want to explore the data using the notebooks:
   ```bash
   jupyter notebook
   ```

### Running Pipeline Commands

The pipeline can be executed using task shortcuts:

*   **Preprocess and Engineer Features**:
    ```bash
    poetry run task preprocess
    ```
*   **Run Code Formatter**:
    ```bash
    poetry run task format
    ```
*   **Run Unit Tests**:
    ```bash
    poetry run task test
    ```

For detailed explanations of parameters and modules, check the source readme in [src/README.md](file:///home/gabyl/projetos/predict-traffic-volume/src/README.md).
