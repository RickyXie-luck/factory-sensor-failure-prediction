# Factory Sensor Failure Prediction

## 1. Project Overview

This project builds a machine learning pipeline to predict whether a factory machine will fail within the next 7 days based on simulated industrial sensor data.

The project covers the full data science workflow, including data loading, missing value analysis, exploratory data analysis, statistical feature screening, feature engineering, model training, model comparison, threshold optimization, and final evaluation.

## 2. Dataset

The dataset used in this project is a factory sensor simulator dataset from Kaggle. It contains machine-level operational information, sensor readings, maintenance records, and failure-related labels.

The dataset contains:

- 500,000 samples
- 22 original columns
- Numerical sensor features, such as temperature, vibration, sound level, oil level, coolant level, power consumption, and operational hours
- Categorical machine type information
- Target variable: `Failure_Within_7_Days`

The target variable is defined as:

- `0`: No failure within 7 days
- `1`: Failure within 7 days

## 3. Project Workflow

The project follows the workflow below:

1. Data loading and basic overview
2. Missing value analysis
3. Exploratory data analysis
4. Statistical feature screening
5. Feature engineering
6. Train-test split
7. Logistic Regression baseline model
8. Random Forest comparison model
9. Model comparison
10. Threshold optimization
11. Feature importance analysis
12. Final conclusion

## 4. Exploratory Data Analysis

During EDA, the following issues and patterns were identified:

- The dataset is highly imbalanced.
- Non-failure samples account for about 94% of the dataset.
- Failure samples account for about 6% of the dataset.
- Several machine-specific columns contain a large number of missing values, including:
  - `Laser_Intensity`
  - `Hydraulic_Pressure_bar`
  - `Coolant_Flow_L_min`
  - `Heat_Index`

These columns were removed before modeling due to their high missing ratios.

## 5. Statistical Feature Screening

To understand which variables are more related to machine failure, several statistical methods were used:

- Group mean comparison between failure and non-failure machines
- Correlation analysis
- Welch's t-test
- Cohen's d effect size

The analysis showed that `Operational_Hours` was the most important variable related to machine failure. Machines that failed within 7 days had much higher average operational hours than non-failure machines.

Other variables such as `Vibration_mms` and `Temperature_C` also showed some relationship with failure, but their effect sizes were much smaller.

## 6. Feature Engineering

The following preprocessing steps were applied:

- Removed ID column: `Machine_ID`
- Removed high-missing-value columns
- Removed `Remaining_Useful_Life_days` to avoid using another target-like variable
- Removed `Failure_Label` to avoid data leakage
- Converted boolean variables into 0/1 values
- Applied one-hot encoding to `Machine_Type`
- Split the dataset into training and test sets using stratified sampling

The final feature matrix contains 46 input features.

## 7. Models

Two machine learning models were trained and compared.

### Logistic Regression

Logistic Regression was used as the baseline model because it is simple, interpretable, and suitable for binary classification.

Class imbalance was handled using `class_weight='balanced'`.

### Random Forest

Random Forest was used as a comparison model to test whether a nonlinear model could improve performance.

## 8. Model Evaluation

Because the dataset is imbalanced, accuracy alone is not sufficient. The project focuses on the following metrics:

- Precision
- Recall
- F1-score
- Confusion Matrix

### Initial Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.928 | 0.455 | 0.972 | 0.620 |
| Random Forest | 0.923 | 0.437 | 0.979 | 0.604 |

The Logistic Regression model achieved a better F1-score and a more balanced performance, while Random Forest achieved slightly higher recall but produced more false positives.

## 9. Threshold Optimization

The initial Logistic Regression model had high recall but relatively low precision, which means it could detect most failure cases but also produced many false alarms.

To reduce false positives, the classification threshold was adjusted from the default value of 0.5 to 0.9.

After threshold optimization:

| Threshold | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| 0.9 | 0.962 | 0.653 | 0.794 | 0.716 |

The optimized threshold improved precision and F1-score significantly, while still maintaining a reasonable recall.

## 10. Key Findings

The main findings are:

1. The dataset is highly imbalanced, so precision, recall, and F1-score are more important than accuracy alone.
2. `Operational_Hours` is the strongest feature associated with machine failure.
3. Logistic Regression performed better than Random Forest in terms of F1-score.
4. Threshold optimization improved model balance by reducing false positives.
5. Data leakage was avoided by removing the target-derived column `Failure_Label`.

## 11. How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle and place the CSV file in the same folder as the notebook.

The expected file name is:

```text
factory_sensor_simulator_2040.csv
```

Then open and run the notebook:

```text
factory_sensor_failure_prediction.ipynb
```

## 12. Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

## 13. Project Highlights

- Completed an end-to-end machine learning workflow on 500,000 industrial sensor records
- Applied EDA, statistical testing, and effect size analysis for feature screening
- Compared Logistic Regression and Random Forest models
- Addressed class imbalance using balanced class weights
- Improved model performance through threshold optimization
- Identified and removed a target-derived feature to avoid data leakage
