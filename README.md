# Kathmandu Daily Climate

## 1. Project Overview

Kathmandu Daily Climate is a machine learning project that analyses historical daily climate data from Kathmandu, Nepal, covering the period from 1982 to 2023.

The project develops and evaluates machine learning regression models to predict the next day's average temperature (`T2M`) using available climate observations and historical temperature information.

Three regression models are implemented and compared:

1. Random Forest Regressor
2. Ridge Regression
3. Lasso Regression

A persistence baseline is also used to assess whether the machine learning models improve upon a simple prediction based on the current day's average temperature.

## 2. Project Objectives

- Explore historical daily climate patterns in Kathmandu.
- Examine relationships between temperature and other climate variables.
- Perform data quality checks and preprocessing.
- Engineer lag, rolling-average, and calendar-based features.
- Develop Random Forest, Ridge, and Lasso regression models.
- Tune model hyperparameters using time-series cross-validation.
- Evaluate models using MAE, MSE, RMSE, and R².
- Compare the models with a persistence baseline.
- Develop an intelligent system architecture for next-day temperature prediction.

## 3. Dataset Information

**Dataset:** Kathmandu Daily Climate Data (1982–2023)

**Dataset link:** [Kathmandu Daily Climate Data – Kaggle](https://www.kaggle.com/datasets/ramkrishnapudasaini/kathmandu-daily-climate-data-1982-2023)

**Original data source:** NASA Langley Research Center (LaRC), Prediction of Worldwide Energy Resources (POWER) Project.

The dataset contains 15,340 daily records and 20 original climate variables covering 1 January 1982 to 31 December 2023.

The data were obtained from the NASA POWER Project through its Data Access Viewer API. They represent NASA POWER meteorological data products rather than direct measurements from a Kathmandu ground-based weather station.

### Dataset Attributes

| Column | Description |
|---|---|
| YEAR | Year of observation |
| DOY | Day of the year |
| T2M | Average temperature at 2 metres |
| T2MWET | Wet-bulb temperature |
| TS | Earth skin temperature |
| T2M_RANGE | Daily temperature range |
| T2M_MAX | Maximum temperature |
| T2M_MIN | Minimum temperature |
| QV2M | Specific humidity |
| RH2M | Relative humidity |
| PRECTOTCORR | Corrected precipitation |
| PS | Surface pressure |
| WS2M | Wind speed at 2 metres |
| WS2M_MAX | Maximum wind speed at 2 metres |
| WS2M_MIN | Minimum wind speed at 2 metres |
| WS2M_RANGE | Wind speed range at 2 metres |
| WS10M | Wind speed at 10 metres |
| WS10M_MAX | Maximum wind speed at 10 metres |
| WS10M_MIN | Minimum wind speed at 10 metres |
| WS10M_RANGE | Wind speed range at 10 metres |

### Data Quality

The initial dataset was checked for missing values, duplicate records, and date continuity.

- Missing values: 0
- Duplicate rows: 0
- Duplicate dates: 0
- Missing calendar days: 0
- Date range: 1982-01-01 to 2023-12-31

## 4. Technologies and Tools

- Python 3.12
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Joblib
- Git and GitHub

## 5. Project Structure

```text
kathmandu-daily-climate/
│
├── data/
│   └── Kathmandu_Daily_Climate_Data.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   └── models/
│       ├── __init__.py
│       ├── random_forest_regressor.py
│       ├── ridge_regressor.py
│       ├── lasso_regressor.py
│       ├── evaluate_models.py
│       └── baseline_model.py
│
├── scripts/
│   └── compare_models.py
│
├── outputs/
│   ├── models/
│   ├── figures/
│   └── results/
│
├── reports/
│   └── figures/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 6. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed using `notebooks/eda.ipynb`.

The analysis included:

- Dataset dimensions, column names, and data types.
- Missing-value and duplicate-record checks.
- Descriptive statistical analysis.
- Daily temperature distribution.
- Monthly average temperature analysis.
- Monthly temperature distribution.
- Annual average temperature variation.
- Temperature correlation with other climate variables.
- Identification of the coldest and warmest recorded days.
- Relationship between current-day and next-day average temperature.

Important findings include:

- Overall mean daily average temperature: approximately 19.20 °C.
- Coldest recorded day: 8 January 2013, at 4.12 °C.
- Warmest recorded day: 10 June 1998, at 30.11 °C.
- Warmest month by average temperature: June, at approximately 25.13 °C.
- Coldest month by average temperature: January, at approximately 11.16 °C.
- Correlation between current-day and next-day average temperature: approximately 0.9864.

The EDA visualisations are saved in `outputs/figures/`.

## 7. Data Preprocessing and Feature Engineering

Data preprocessing is implemented in:

```text
src/preprocess.py
```

The preprocessing workflow includes:

1. Loading the original CSV dataset.
2. Constructing a date column using `YEAR` and `DOY`.
3. Sorting observations chronologically.
4. Selecting the climate variables used as input features.
5. Creating temperature lag features.
6. Calculating rolling temperature averages.
7. Creating next-day seasonal calendar features.
8. Shifting the target variable to represent the following day's average temperature.
9. Removing rows that cannot be used because of unavailable lag, rolling, or target values.
10. Dividing the data chronologically into training, validation, and testing sets.

### Target Variable

The prediction target is the next day's average temperature (`T2M`).

The target is created by shifting the original temperature series by one day.

The prediction is assumed to be made after the current day's climate observations are available. Therefore, the model uses information from the current day and earlier observations, not climate measurements from the future prediction day.

### Engineered Features

The feature set contains 25 input features, including:

- Original daily climate variables.
- Temperature lag features: `T2M_LAG_1`, `T2M_LAG_2`, and `T2M_LAG_3`.
- Rolling averages: `T2M_ROLLING_3` and `T2M_ROLLING_7`.
- Next-day seasonal calendar features: `NEXT_DAY_DOY_SIN` and `NEXT_DAY_DOY_COS`.

`T2M_LAG_1` represents the current day's temperature and duplicates the original `T2M` feature. This redundancy is acknowledged as a limitation of the current feature set.

### Data Splitting

The processed dataset contains 15,333 usable observations.

The chronological split is:

| Dataset | Records |
|---|---:|
| Training | 10,732 |
| Validation | 2,299 |
| Testing | 2,300 |

A small gap is maintained at the split boundaries to avoid overlap between target dates.

The test set is kept separate from hyperparameter tuning and model selection.

## 8. Machine Learning Models

Three regression algorithms were implemented.

### 8.1 Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

It can capture non-linear relationships between climate variables and next-day temperature.

The model was tuned using parameters such as:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

Implementation:

```text
src/models/random_forest_regressor.py
```

### 8.2 Ridge Regression

Ridge Regression is a linear regression algorithm that uses L2 regularisation to penalise large coefficient values.

Feature scaling was applied using `StandardScaler` within a Scikit-learn pipeline.

The regularisation parameter `alpha` was tuned using cross-validation.

Implementation:

```text
src/models/ridge_regressor.py
```

### 8.3 Lasso Regression

Lasso Regression uses L1 regularisation to reduce coefficient values and can set some coefficients to zero.

Feature scaling was applied using `StandardScaler` within a pipeline. The model was configured with an increased iteration limit to support convergence.

The regularisation parameter `alpha` was tuned using cross-validation.

Implementation:

```text
src/models/lasso_regressor.py
```

## 9. Hyperparameter Tuning

Hyperparameter tuning was conducted using:

- `GridSearchCV`
- `TimeSeriesSplit` with five splits and a one-observation gap.
- `neg_root_mean_squared_error` as the scoring metric.

The best hyperparameters were selected using training data and time-series cross-validation.

The validation dataset was used to compare model performance. The final test dataset remained separate until the final evaluation.

### Selected Hyperparameters

| Model | Selected parameters |
|---|---|
| Random Forest | 200 estimators, unlimited maximum depth, `sqrt` maximum features, minimum leaf size 2, minimum split size 5 |
| Ridge | alpha = 0.1 |
| Lasso | alpha = 0.0001 |

## 10. Model Evaluation

All three models were evaluated on the same unseen chronological test set.

The evaluation metrics were:

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |

A persistence baseline was also evaluated. It predicts the next day's average temperature using the current day's average temperature.

The baseline formula is:

\[
\hat{T}_{t+1}=T_t
\]

### Final Test Results

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Random Forest | 0.580843 | 0.597414 | 0.772926 | 0.973863 |
| Lasso | 0.580778 | 0.600365 | 0.774832 | 0.973734 |
| Ridge | 0.580909 | 0.601052 | 0.775275 | 0.973704 |
| Persistence Baseline | 0.617004 | 0.672477 | 0.820047 | 0.970579 |

Random Forest achieved the lowest RMSE and highest R² among the three machine learning models on the test set.

Lasso achieved a marginally lower MAE. The performance differences between the three machine learning models were small.

All three machine learning models outperformed the persistence baseline on RMSE.

The high R² values should be interpreted carefully because daily temperatures are strongly correlated across consecutive days. The persistence baseline provides useful context for understanding this effect.

## 11. Model Comparison

The final model comparison is generated using:

```text
scripts/compare_models.py
```

The comparison combines the test results of the three machine learning models with the persistence baseline.

The comparison results are saved in:

```text
outputs/results/final_model_comparison.csv
```

The final RMSE comparison chart is saved in:

```text
outputs/figures/final_rmse_comparison.png
```

## 12. Intelligent System Architecture

The proposed system consists of five main components:

1. **Data Acquisition:** Obtains historical daily climate data.
2. **Data Preprocessing and Feature Engineering:** Constructs dates, prepares input features, and generates lag, rolling-average, and calendar features.
3. **Machine Learning Engine:** Trains Random Forest, Ridge, and Lasso regression models.
4. **Model Evaluation and Comparison:** Uses time-series cross-validation, validation results, test metrics, and a persistence baseline to assess performance.
5. **Prediction Output:** Produces a next-day average temperature estimate using the selected model.

The complete system architecture and data flow are documented in the project report.

## 13. Ethical Considerations and Governance

The project considers the following ethical and governance issues:

- **Data quality:** Dataset quality checks were performed to identify missing values, duplicate records, and date inconsistencies.
- **Transparency:** The models, input features, and evaluation metrics are documented.
- **Privacy:** The project uses climate data and does not require personal information.
- **Reliability:** Model predictions may contain errors and should not be treated as guaranteed outcomes.
- **Responsible use:** The predictions are intended for academic analysis and should not replace official weather forecasts or emergency weather warnings.
- **Data limitations:** NASA POWER meteorological products are not equivalent to direct measurements from a local weather station.

## 14. Running the Project

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd kathmandu-daily-climate
```

### Step 2: Create and activate a virtual environment

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add the dataset

Download the CSV dataset from Kaggle and place it at:

```text
data/Kathmandu_Daily_Climate_Data.csv
```

### Step 5: Run preprocessing

```bash
python -m src.preprocess
```

### Step 6: Run the models

```bash
python -m src.models.random_forest_regressor
python -m src.models.ridge_regressor
python -m src.models.lasso_regressor
```

### Step 7: Evaluate the models

```bash
python -m src.models.evaluate_models
```

### Step 8: Run the baseline

```bash
python -m src.models.baseline_model
```

### Step 9: Generate the final comparison

```bash
python -m scripts.compare_models
```

### Step 10: Open the EDA notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/eda.ipynb
```

## 15. Output Files

The project stores generated artifacts in the `outputs/` directory.

- `outputs/models/` contains saved trained models.
- `outputs/results/` contains evaluation metrics, hyperparameter search histories, and model comparison results.
- `outputs/figures/` contains EDA and model evaluation visualisations.

## 16. Dataset Acknowledgement

These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program.

The dataset is distributed through Kaggle, while NASA POWER is acknowledged as the original data provider.

## 17. Conclusion

The Kathmandu Daily Climate project developed and evaluated three machine learning regression models for next-day average temperature prediction.

The results show that Random Forest achieved the best test RMSE among the machine learning models, while Lasso achieved a marginally lower MAE. All three models performed better than the persistence baseline in terms of RMSE.

The project demonstrates an end-to-end machine learning workflow involving climate data exploration, preprocessing, feature engineering, time-series hyperparameter tuning, model evaluation, and comparison.

The findings should be interpreted in the context of the dataset's temporal characteristics, the strong relationship between consecutive daily temperatures, and the limitations of gridded meteorological data.
