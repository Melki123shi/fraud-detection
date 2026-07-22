# Interim-1 Submission: Data Analysis and Preprocessing

**Project:** Improved Detection of Fraud Cases for E-commerce and Bank Transactions
**Team:** Adey Innovations Inc.
**Submission Date:** 07 Jun 2026

---

## 1. Overview

This report documents the data analysis and preprocessing work completed for Task 1 of the fraud detection project. The objective is to prepare clean, feature-rich datasets ready for modeling across two transaction streams: e-commerce transactions (`Fraud_Data.csv`) and bank credit card transactions (`creditcard.csv`).

---

## 2. Data Cleaning

### 2.1 Fraud_Data.csv (E-commerce Transactions)

- **Raw shape:** 151,112 rows x 11 columns
- **Missing values:** None found across all columns
- **Duplicates:** 0 exact duplicate rows detected
- **Data types corrected:**
  - `signup_time` and `purchase_time` converted from string to `datetime64[ns]`
  - `ip_address` stored as float64 (converted to int64 during geolocation mapping)

### 2.2 creditcard.csv (Bank Credit Card Transactions)

- **Raw shape:** 284,807 rows x 31 columns
- **Missing values:** None found across all 31 columns
- **Duplicates:** 1,081 duplicate rows (0.38%) detected and removed
- **Cleaned shape:** 283,726 rows x 31 columns
- **Data types:** All columns already numeric (float64 for V1-V28, Time, Amount; int64 for Class)

### 2.3 IpAddress_to_Country.csv

- **Records:** ~138,000 IP ranges
- **Columns:** `lower_bound_ip_address`, `upper_bound_ip_address`, `country`
- **No cleaning required** — used as-is for range-based lookup

**Justification:** Both datasets were free of missing values. Duplicates in the credit card dataset were removed to prevent inflated counts. The fraud dataset had no duplicates, confirming data integrity.

---

## 3. Exploratory Data Analysis

### 3.1 Fraud_Data.csv — Key Findings

#### Class Imbalance
- **Legitimate (0):** 136,961 (90.64%)
- **Fraud (1):** 14,151 (9.36%)
- **Imbalance ratio:** 9.7:1

#### Univariate Analysis
- **Purchase value:** Right-skewed distribution with mean ~$36.93. Fraudulent and legitimate transactions show similar distributions, indicating purchase value alone is not a strong discriminator.
- **Age:** Normally distributed around 33 years. No significant difference between fraud and legitimate classes.
- **Categorical features:** Chrome is the most used browser (40.6%), SEO is the top source (40.1%), and males account for 58.4% of transactions.

#### Bivariate Analysis
- **Source:** Direct traffic has the highest fraud rate (10.54%), followed by Ads (9.21%) and SEO (8.93%).
- **Browser:** Chrome shows the highest fraud rate (9.88%), though differences across browsers are modest.
- **Sex:** Males have a slightly higher fraud rate (9.55%) than females (9.10%).
- **Age groups:** Fraud rates are relatively consistent across age groups, with a slight elevation in the 18-25 bracket.
- **Purchase value ranges:** Transactions in the $0-20 and $51-100 ranges show higher fraud rates (~9.8%) compared to $21-50 (~9.1%).

#### Temporal Analysis
- **Hour of day:** Fraud rates peak during early morning hours (0-5 AM) and show variation throughout the day.
- **Day of week:** Relatively consistent fraud rates across days.
- **Time since signup:** This is a **critical finding** — fraudulent transactions have a median `time_since_signup` of ~0.0003 hours (~1 second), compared to ~1,443 hours (~60 days) for legitimate transactions. This indicates that most fraudulent transactions occur almost immediately after account creation.

#### Device and User Analysis
- 6,175 devices were used for multiple transactions.
- Several devices show 100% fraud rates with 3+ transactions, suggesting device reuse by fraudsters.
- Each user ID appears exactly once in the dataset (one transaction per user).

### 3.2 creditcard.csv — Key Findings

#### Class Imbalance
- **Legitimate (0):** 283,253 (99.83%)
- **Fraud (1):** 473 (0.17%)
- **Imbalance ratio:** 598.8:1 (extreme imbalance)

#### Amount Analysis
- Heavily right-skewed (mean $88.41, median $22.00)
- Fraudulent transactions have a higher mean ($123.87) but lower median ($9.82), indicating a bimodal pattern
- Log transformation (`log1p(Amount)`) helps normalize the distribution

#### Time Analysis
- Time is measured in seconds since the first transaction
- Fraudulent transactions tend to occur earlier (mean ~80,451s) compared to legitimate (~94,835s)
- Time alone is not a strong fraud predictor

#### PCA Features (V1-V28)
- Features with the **largest mean differences** between classes: V14, V3, V17, V12, V10
- These features show clear class separation and are expected to be the most predictive
- Correlation analysis confirms V17 (-0.31), V14 (-0.29), V12 (-0.25), V10 (-0.21) as the most negatively correlated with fraud

#### Outliers
- 31,685 Amount outliers (11.17%) identified using IQR method
- Fraud rate in outliers (0.27%) is higher than in normal transactions (0.15%)

---

## 4. Geolocation Integration

### 4.1 Method

IP addresses were converted from float to integer format. A **binary search algorithm** was implemented to efficiently match each transaction's IP address against the IP range database (`IpAddress_to_Country.csv`). For each transaction, the algorithm finds which country range contains its IP address.

### 4.2 Results

- **182 unique countries** mapped from 151,112 transactions
- **Top 5 countries by volume:** United States (58,049), Unknown (21,966), China (12,038), Japan (7,306), United Kingdom (4,490)
- **21,966 transactions (14.5%)** could not be mapped to a country (labeled "Unknown")

### 4.3 Country Risk Features

Two binary features were engineered:
- `is_high_risk_country`: Flag for transactions from the top 10 countries by fraud rate (minimum 100 transactions)
- `is_unknown_country`: Flag for transactions with unmapped IP addresses

**Key insight:** High-risk country transactions account for only 1.57% of all transactions but have an **18.85% fraud rate** — nearly double the overall average of 9.36%. This confirms that geolocation is a meaningful fraud signal.

---

## 5. Feature Engineering

### 5.1 Temporal Features (Fraud_Data.csv)

| Feature | Description |
|---------|-------------|
| `signup_hour` | Hour of day for signup |
| `signup_day` | Day of week for signup |
| `signup_month` | Month of signup |
| `signup_is_weekend` | Binary flag for weekend signup |
| `purchase_hour` | Hour of day for purchase |
| `purchase_day` | Day of week for purchase |
| `purchase_month` | Month of purchase |
| `purchase_is_weekend` | Binary flag for weekend purchase |
| `time_since_signup` | Duration between signup and purchase (in hours) |
| `hour_of_day` | Alias for purchase_hour |
| `day_of_week` | Alias for purchase_day |
| `time_of_day` | Categorical: night (0-6), morning (6-12), afternoon (12-18), evening (18-24) |

### 5.2 Transaction Velocity Features (Fraud_Data.csv)

| Feature | Description |
|---------|-------------|
| `user_total_transactions` | Total transactions per user |
| `device_total_transactions` | Total transactions per device |
| `user_historical_fraud_rate` | Historical fraud rate per user |
| `device_historical_fraud_rate` | Historical fraud rate per device |
| `users_per_device` | Number of unique users per device |
| `devices_per_user` | Number of unique devices per user |

### 5.3 Feature Engineering (creditcard.csv)

| Feature | Description |
|---------|-------------|
| `Amount_log` | `log1p(Amount)` — log-transformed transaction amount |
| `Time_hours` | Time converted from seconds to hours |

### 5.4 Categorical Encoding

One-hot encoding applied to:
- `source` ( Ads, Direct, SEO → drop_first=True)
- `browser` (Chrome, Firefox, IE, Opera, Safari → drop_first=True)
- `sex` (F, M → drop_first=True)
- `time_of_day` (afternoon, evening, morning, night → drop_first=True)

---

## 6. Data Transformation

### 6.1 Scaling

**StandardScaler** was applied to numerical features in both datasets:

**Fraud_Data.csv numerical features scaled:**
- `purchase_value`, `age`, `time_since_signup`
- `user_total_transactions`, `device_total_transactions`
- `user_historical_fraud_rate`, `device_historical_fraud_rate`
- `users_per_device`, `devices_per_user`
- `signup_hour`, `purchase_hour`, `signup_month`, `purchase_month`

**creditcard.csv numerical features scaled:**
- `Amount_log`, `Time_hours`
- V1-V28 were already PCA-transformed and standardized

### 6.2 Train-Test Split

Stratified 80/20 split preserving class distribution:

| Dataset | Train | Test |
|---------|-------|------|
| Fraud_Data.csv | 120,889 | 30,223 |
| creditcard.csv | 226,980 | 56,746 |

---

## 7. Class Imbalance Handling

### 7.1 Strategy: SMOTE

**SMOTE (Synthetic Minority Over-sampling Technique)** was chosen over undersampling for the following reasons:

1. **Preserves information:** Undersampling discards potentially useful majority class samples. With 151K+ fraud records and 284K+ credit records, discarding data would be wasteful.
2. **Creates synthetic samples:** SMOTE generates new minority samples by interpolating between existing ones, rather than simply duplicating records (which would cause overfitting).
3. **Moderate oversampling ratios:** The ratios chosen avoid excessive synthetic sample generation while meaningfully addressing imbalance.

### 7.2 Before and After Resampling

**Fraud_Data.csv (SMOTE ratio = 0.5, target 1:2):**

| | Before SMOTE | After SMOTE |
|---|---|---|
| Class 0 (Legitimate) | 109,568 (90.64%) | 109,568 (66.67%) |
| Class 1 (Fraud) | 11,321 (9.36%) | 54,784 (33.33%) |
| Imbalance Ratio | 9.7:1 | 2.0:1 |
| Total | 120,889 | 164,352 |

**creditcard.csv (SMOTE ratio = 0.3, target ~1:3):**

| | Before SMOTE | After SMOTE |
|---|---|---|
| Class 0 (Legitimate) | 226,602 (99.83%) | 226,602 (76.92%) |
| Class 1 (Fraud) | 378 (0.17%) | 67,980 (23.08%) |
| Imbalance Ratio | 599.5:1 | 3.3:1 |
| Total | 226,980 | 294,582 |

### 7.3 Justification of Ratios

- **Fraud_Data.csv (1:2):** The original 9.7:1 imbalance is moderate. A 1:2 ratio provides enough minority samples for the model to learn fraud patterns without introducing excessive synthetic noise.
- **creditcard.csv (1:3):** The extreme 599.5:1 imbalance requires more aggressive oversampling. A 1:3 ratio (vs 1:2 for fraud data) ensures the model sees sufficient fraud examples, though the larger ratio introduces more synthetic samples. This trade-off is necessary given the extreme scarcity of fraud cases (only 473 in the full dataset).

### 7.4 Critical: Data Leakage Prevention

SMOTE was applied **only to training data** after the train-test split. The test sets remain untouched with their original class distributions, ensuring unbiased evaluation.

---

## 8. Processed Datasets

All processed data is saved to `data/processed/`:

| File | Description |
|------|-------------|
| `fraud_data_features.csv` | Fully processed Fraud_Data.csv with all engineered features |
| `creditcard_features.csv` | Fully processed creditcard.csv with engineered features |
| `X_train_fraud.csv` | Training features (Fraud) |
| `X_test_fraud.csv` | Test features (Fraud) |
| `y_train_fraud.csv` | Training labels (Fraud) |
| `y_test_fraud.csv` | Test labels (Fraud) |
| `X_train_fraud_smote.csv` | SMOTE-resampled training features (Fraud) |
| `y_train_fraud_smote.csv` | SMOTE-resampled training labels (Fraud) |
| `X_train_credit.csv` | Training features (Credit) |
| `X_test_credit.csv` | Test features (Credit) |
| `y_train_credit.csv` | Training labels (Credit) |
| `y_test_credit.csv` | Test labels (Credit) |
| `X_train_credit_smote.csv` | SMOTE-resampled training features (Credit) |
| `y_train_credit_smote.csv` | SMOTE-resampled training labels (Credit) |

---

## 9. Summary of Deliverables

| Deliverable | Status | Location |
|-------------|--------|----------|
| Cleaned datasets | Complete | `data/processed/` |
| EDA report (Fraud_Data.csv) | Complete | `notebooks/eda-fraud-data.ipynb` |
| EDA report (creditcard.csv) | Complete | `notebooks/eda-creditcard.ipynb` |
| Feature engineering documentation | Complete | `notebooks/feature-engineering.ipynb`, `src/data_processing.py` |
| Geolocation integration | Complete | `src/data_processing.py` (binary search lookup) |
| Class imbalance handling strategy | Complete | `notebooks/feature-engineering.ipynb`, this report |
| Resampling justification | Complete | Section 7 of this report |

---

## 10. Key Insights for Modeling (Task 2)

Based on the EDA and feature engineering work, the following insights will guide model building:

1. **`time_since_signup`** is the strongest individual fraud signal — fraudulent transactions occur almost immediately after signup.
2. **Geolocation features** (`is_high_risk_country`) provide meaningful discrimination, with high-risk countries showing 2x the fraud rate.
3. **Transaction velocity features** (`user_total_transactions`, `device_total_transactions`) capture behavioral patterns associated with fraud.
4. **PCA features V14, V17, V12, V10** are the most predictive in the credit card dataset.
5. **Extreme class imbalance** (especially in creditcard.csv at 0.17% fraud) necessitates metrics beyond accuracy — AUC-PR, F1-score, and confusion matrices will be essential.
6. **SMOTE on training data only** ensures valid evaluation on untouched test sets.

---

## 11. Testing

### 11.1 Test Infrastructure

- **Framework:** pytest
- **Test file:** `tests/test_data_processing.py`
- **CI command:** `pytest tests/ -v` (run via `.github/workflows/ci.yml`)
- **Linting:** `flake8 src/ tests/` enforced in CI

### 11.2 Test Coverage

59 tests across 17 test classes covering all functions in `src/data_processing.py`:

| Category | Functions | Tests |
|----------|-----------|-------|
| Utilities | `_ensure_dir` | 2 |
| Loading | `load_data` | 2 |
| Basic Checks | `check_missing_values`, `check_duplicates`, `check_outliers` | 8 |
| Cleaning | `correct_data_types`, `handle_cleaning`, `save_cleaned_data` | 6 |
| IP Mapping | `map_ip_to_country`, `prepare_ip_ranges`, `add_ip_country`, `add_country_risk_features`, `get_fraud_by_country` | 10 |
| Temporal Features | `categorize_hour`, `engineer_temporal_features` | 6 |
| Velocity Features | `engineer_velocity_features` | 5 |
| EDA Helpers | `get_class_distribution`, `get_fraud_rate_by_column` | 4 |
| Encoding | `encode_categoricals` | 3 |
| Scaling & Splitting | `scale_features`, `stratified_split`, `apply_smote` | 5 |

### 11.3 Running Tests Locally

```bash
# Run all tests
pytest tests/ -v

# Run tests for a specific class
pytest tests/test_data_processing.py::TestCategorizeHour -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Task 2 Documentation

For Task 2 (Model Building, Training, and Evaluation), see **[notebooks/doc.md](doc.md)** for the full blog-style implementation report, model comparison results, and evaluation insights.
