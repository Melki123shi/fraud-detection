# Fraud Detection for E-commerce and Bank Transactions

A comprehensive fraud detection system for Adey Innovations Inc., a leading FinTech company serving e-commerce and banking clients. This project analyzes and builds classification models for fraud detection across two transaction streams: e-commerce transactions and bank credit card transactions.

## Project Structure

```
fraud-detection/
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/
│   ├── raw/                    # Original datasets
│   │   ├── Fraud_Data.csv
│   │   ├── IpAddress_to_Country.csv
│   │   └── creditcard.csv
│   └── processed/              # Cleaned and feature-engineered data
├── notebooks/
│   ├── __init__.py
│   ├── eda-fraud-data.ipynb    # EDA for e-commerce fraud data
│   ├── eda-creditcard.ipynb    # EDA for credit card fraud data
│   ├── feature-engineering.ipynb # Feature engineering and data transformation
│   ├── modeling.ipynb          # Model building and evaluation
│   ├── shap-explainability.ipynb # SHAP analysis
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── constants.py            # Project constants
│   └── data_processing.py      # Data processing utilities
├── tests/
│   └── __init__.py
├── models/                     # Saved model artifacts
├── scripts/
│   ├── __init__.py
│   └── README.md
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebooks

The project is organized into the following notebooks:

1. **EDA for Fraud Data** (`notebooks/eda-fraud-data.ipynb`)
   - Data loading and inspection
   - Data cleaning (missing values, duplicates, data types)
   - Univariate and bivariate analysis
   - Class imbalance analysis

2. **EDA for Credit Card Data** (`notebooks/eda-creditcard.ipynb`)
   - Data loading and inspection
   - Analysis of PCA-transformed features
   - Amount and Time feature analysis
   - Class imbalance analysis

3. **Feature Engineering** (`notebooks/feature-engineering.ipynb`)
   - Geolocation integration (IP-to-country mapping)
   - Temporal feature engineering
   - Transaction velocity features
   - Data transformation and scaling
   - SMOTE for class imbalance handling

4. **Modeling** (`notebooks/modeling.ipynb`) - *Coming in Interim-2*
   - Logistic Regression baseline
   - Random Forest / XGBoost ensemble models
   - Model evaluation and comparison

5. **SHAP Explainability** (`notebooks/shap-explainability.ipynb`) - *Coming in Final*
   - SHAP summary plots
   - SHAP force plots
   - Business recommendations

## Datasets

### Fraud_Data.csv (E-commerce Transactions)
- **Records**: ~151,000
- **Features**: User ID, signup/purchase times, purchase value, device ID, source, browser, sex, age, IP address
- **Target**: `class` (1 = fraud, 0 = legitimate)
- **Class Distribution**: ~91% legitimate, ~9% fraud

### creditcard.csv (Bank Credit Card Transactions)
- **Records**: ~284,000
- **Features**: Time, V1-V28 (PCA-transformed), Amount
- **Target**: `Class` (1 = fraud, 0 = legitimate)
- **Class Distribution**: ~99.83% legitimate, ~0.17% fraud

### IpAddress_to_Country.csv
- **Records**: ~138,000
- **Features**: IP address ranges (lower/upper bounds) and corresponding country

## Task 1: Data Analysis and Preprocessing

### Data Cleaning
- **Missing Values**: None found in either dataset
- **Duplicates**: Removed duplicate rows
- **Data Types**: Converted timestamps to datetime, IP addresses to integers

### Geolocation Integration
- Mapped IP addresses to countries using range-based lookup with binary search
- Created country-based features:
  - `country`: Country corresponding to IP address
  - `is_high_risk_country`: Flag for top 10 countries by fraud rate
  - `is_unknown_country`: Flag for unmatched IP addresses

### Feature Engineering (Fraud_Data.csv)
1. **Temporal Features**:
   - `signup_hour`, `purchase_hour`: Hour of day
   - `signup_day`, `purchase_day`: Day of week
   - `signup_month`, `purchase_month`: Month
   - `signup_is_weekend`, `purchase_is_weekend`: Weekend flags
   - `time_since_signup`: Duration between signup and purchase (hours)
   - `time_of_day`: Categorical (night, morning, afternoon, evening)

2. **Transaction Velocity Features**:
   - `user_total_transactions`: Total transactions per user
   - `device_total_transactions`: Total transactions per device
   - `user_historical_fraud_rate`: User's historical fraud rate
   - `device_historical_fraud_rate`: Device's historical fraud rate
   - `users_per_device`: Unique users per device
   - `devices_per_user`: Unique devices per user

3. **Categorical Encoding**:
   - One-hot encoded: source, browser, sex, time_of_day

### Data Transformation
- **Scaling**: StandardScaler applied to numerical features
- **Log Transformation**: Applied to Amount in creditcard.csv to reduce skewness

### Class Imbalance Handling
- **SMOTE (Synthetic Minority Over-sampling Technique)**
- **Fraud_Data.csv**: 1:2 ratio (original ~1:10)
- **creditcard.csv**: 1:3 ratio (original ~1:588)
- Applied only to training sets to prevent data leakage

## Key Findings

### Fraud_Data.csv
1. **Time-based patterns**: Fraud rates vary significantly by hour of day
2. **Purchase value**: Certain value ranges show higher fraud rates
3. **Browser differences**: Some browsers have higher fraud rates
4. **Time since signup**: Fraudulent transactions often occur quickly after signup
5. **Geographic patterns**: Fraud rates vary by country

### creditcard.csv
1. **Extreme imbalance**: Only 0.17% of transactions are fraudulent
2. **PCA features**: V4, V11, V12, V14, V17 show clear separation between classes
3. **Amount**: Fraud transactions have different amount patterns
4. **Time**: No strong temporal patterns observed

## Metrics

The project uses metrics appropriate for imbalanced data:
- **AUC-PR** (Area Under Precision-Recall Curve)
- **F1-Score** (Harmonic mean of precision and recall)
- **Confusion Matrix** (TP, FP, TN, FN)
- **Precision** and **Recall**

## Technologies Used

- Python 3.11
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- imbalanced-learn (SMOTE)
- SHAP (for explainability)

## License

This project is for educational purposes as part of the FinTech program.

## Acknowledgments

- Adey Innovations Inc. for the project context
- Course tutors: Kerod, Mahbubah, Feven
