import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def load_fraud_data():
    return load_data(DATA_RAW / "Fraud_Data.csv")


def load_creditcard_data():
    return load_data(DATA_RAW / "creditcard.csv")


def load_ip_to_country():
    return load_data(DATA_RAW / "IpAddress_to_Country.csv")


# ---------------------------------------------------------------------------
# Basic checks
# ---------------------------------------------------------------------------
def check_missing_values(data):
    if data is not None:
        return data.isnull().sum()
    print("Data is None.")
    return None


def check_duplicates(data):
    if data is not None:
        return data[data.duplicated()]
    print("Data is None.")
    return None


def check_outliers(data, column):
    if data is not None and column in data.columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return data[(data[column] < lower) | (data[column] > upper)]
    return None


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def correct_data_types(data, column_types):
    if data is not None:
        for col, dtype in column_types.items():
            if col in data.columns:
                data[col] = data[col].astype(dtype)
    return data


def handle_cleaning(data):
    if data is None:
        print("Data is None.")
        return None
    print("Missing values:\n", check_missing_values(data))
    print("Duplicate rows:", len(check_duplicates(data)))
    data = data.drop_duplicates().reset_index(drop=True)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    return data


def save_cleaned_data(data, file_path):
    if data is not None:
        try:
            data.to_csv(file_path, index=False)
            print(f"Saved to {file_path}")
        except Exception as e:
            print(f"Error saving: {e}")


# ---------------------------------------------------------------------------
# IP-to-country mapping
# ---------------------------------------------------------------------------
def map_ip_to_country(ip_int, ip_ranges):
    left, right = 0, len(ip_ranges) - 1
    while left <= right:
        mid = (left + right) // 2
        if ip_int < ip_ranges.iloc[mid]["lower_bound_ip_address"]:
            right = mid - 1
        elif ip_int > ip_ranges.iloc[mid]["upper_bound_ip_address"]:
            left = mid + 1
        else:
            return ip_ranges.iloc[mid]["country"]
    return "Unknown"


def prepare_ip_ranges(ip_df):
    df = ip_df.copy()
    df["lower_bound_ip_address"] = df["lower_bound_ip_address"].astype(int)
    df["upper_bound_ip_address"] = df["upper_bound_ip_address"].astype(int)
    return df.sort_values("lower_bound_ip_address").reset_index(drop=True)


def add_ip_country(df, ip_ranges):
    df = df.copy()
    df["ip_address_int"] = df["ip_address"].astype(int)
    df["country"] = df["ip_address_int"].apply(
        lambda x: map_ip_to_country(x, ip_ranges)
    )
    return df


def add_country_risk_features(df, min_transactions=100, top_n=10):
    stats = df.groupby("country")["class"].agg(["mean", "count"])
    stats.columns = ["fraud_rate", "total_transactions"]
    significant = stats[stats["total_transactions"] >= min_transactions]
    significant_sorted_value = significant.sort_values(
        "fraud_rate",
        ascending=False,
    )
    high_risk = significant_sorted_value.head(top_n).index.tolist()
    df = df.copy()
    df["is_high_risk_country"] = df["country"].isin(high_risk).astype(int)
    df["is_unknown_country"] = (df["country"] == "Unknown").astype(int)
    return df, significant


def get_fraud_by_country(df, min_transactions=100):
    stats = df.groupby("country")["class"].agg(["mean", "count"])
    stats.columns = ["fraud_rate", "total_transactions"]
    stats["fraud_rate"] *= 100
    return stats[stats["total_transactions"] >= min_transactions].sort_values(
        "fraud_rate", ascending=False
    )


def get_country_distribution(df):
    country_counts = df["country"].value_counts().reset_index()
    country_counts.columns = ["country", "count"]
    return country_counts


def get_fraud_prob_by_country_count(df, min_transactions=100):
    stats = df.groupby("country")["class"].agg(["mean", "count"])
    stats.columns = ["fraud_rate", "total_transactions"]
    stats = stats[stats["total_transactions"] >= min_transactions]
    stats["fraud_rate"] *= 100
    return stats.sort_values("total_transactions", ascending=False)


def get_fraud_prob_by_ip_sharing(df, target_col="class"):
    ip_stats = df.groupby("ip_address")[target_col].agg(["mean", "count"])
    ip_stats.columns = ["fraud_rate", "sharing_count"]
    ip_stats["fraud_rate"] *= 100
    sharing_agg = ip_stats.groupby("sharing_count")["fraud_rate"].agg(
        ["mean", "count"]
    )
    sharing_agg.columns = ["avg_fraud_rate", "num_ips"]
    sharing_agg.index.name = "ip_sharing_count"
    return sharing_agg.sort_index()


def get_fraud_prob_by_device_sharing(
    df, device_col="device_id", target_col="class"
):
    device_stats = df.groupby(device_col)[target_col].agg(["mean", "count"])
    device_stats.columns = ["fraud_rate", "sharing_count"]
    device_stats["fraud_rate"] *= 100
    sharing_agg = device_stats.groupby("sharing_count")["fraud_rate"].agg(
        ["mean", "count"]
    )
    sharing_agg.columns = ["avg_fraud_rate", "num_devices"]
    sharing_agg.index.name = "device_sharing_count"
    return sharing_agg.sort_index()


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------
def categorize_hour(hour):
    if 0 <= hour < 6:
        return "night"
    elif 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    return "evening"


def engineer_temporal_features(
    df, signup_col="signup_time", purchase_col="purchase_time"
):
    df = df.copy()
    df["signup_hour"] = df[signup_col].dt.hour
    df["signup_day"] = df[signup_col].dt.day_name()
    df["signup_month"] = df[signup_col].dt.month
    is_weekend = df[signup_col].dt.dayofweek.isin([5, 6])
    df["signup_is_weekend"] = is_weekend.astype(int)

    df["purchase_hour"] = df[purchase_col].dt.hour
    df["purchase_day"] = df[purchase_col].dt.day_name()
    df["purchase_month"] = df[purchase_col].dt.month
    is_weekend = df[purchase_col].dt.dayofweek.isin([5, 6])
    df["purchase_is_weekend"] = is_weekend.astype(int)

    df["time_since_signup"] = (
        df[purchase_col] - df[signup_col]
    ).dt.total_seconds() / 3600
    df["hour_of_day"] = df[purchase_col].dt.hour
    df["day_of_week"] = df[purchase_col].dt.day_name()
    df["time_of_day"] = df["hour_of_day"].apply(categorize_hour)
    return df


# ---------------------------------------------------------------------------
# Velocity features
# ---------------------------------------------------------------------------
def engineer_velocity_features(
    df, user_col="user_id", device_col="device_id", target_col="class"
):
    df = df.copy()
    df["user_total_transactions"] = (
        df.groupby(user_col)[user_col].transform("count")
    )
    df["device_total_transactions"] = (
        df.groupby(device_col)[device_col].transform("count")
    )
    df["user_historical_fraud_rate"] = (
        df.groupby(user_col)[target_col].transform("mean")
    )
    df["device_historical_fraud_rate"] = (
        df.groupby(device_col)[target_col].transform("mean")
    )
    df_group_device = df.groupby(device_col)[user_col]
    df_group_user = df.groupby(user_col)[device_col]
    df["users_per_device"] = df_group_device.transform("nunique")
    df["devices_per_user"] = df_group_user.transform("nunique")

    if "purchase_time" in df.columns:
        df = df.sort_values([user_col, "purchase_time"])
        for window, label in [(24, "24h"), (7, "7d"), (30, "30d")]:
            if label.endswith("h"):
                delta = pd.Timedelta(hours=window)
            else:
                delta = pd.Timedelta(days=window)
            df[f"user_transactions_{label}"] = (
                df.set_index("purchase_time")
                .groupby(user_col)[user_col]
                .rolling(delta)
                .count()
                .values
            )
        df = df.sort_index()

    return df


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------
def get_class_distribution(y, name=""):
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    dist = {}
    print(f"\nClass Distribution {name}:")
    print("=" * 50)
    for cls, count in zip(unique, counts):
        pct = (count / total) * 100
        dist[cls] = {"count": count, "percentage": pct}
        print(f"  Class {cls}: {count:,} ({pct:.2f}%)")
    if len(unique) == 2:
        print(f"  Imbalance Ratio: {counts[0] / counts[1]:.1f}:1")
    return dist


def get_fraud_rate_by_column(df, col, target="class"):
    stats = df.groupby(col)[target].agg(["mean", "count"])
    stats.columns = ["fraud_rate", "total_transactions"]
    stats["fraud_rate"] *= 100
    return stats.sort_values("fraud_rate", ascending=False)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def encode_categoricals(df, columns, drop_first=True):
    return pd.get_dummies(df, columns=columns, drop_first=drop_first)


# ---------------------------------------------------------------------------
# Scaling & splitting
# ---------------------------------------------------------------------------
def scale_features(X_train, X_test, columns):
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns] = scaler.transform(X_test[columns])
    return X_train, X_test, scaler


def stratified_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def apply_smote(X_train, y_train, sampling_strategy=0.5, random_state=42):
    smote = SMOTE(
        random_state=random_state,
        sampling_strategy=sampling_strategy,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


# ---------------------------------------------------------------------------
# Full processing pipelines
# ---------------------------------------------------------------------------
def process_fraud_data():
    print("=" * 60)
    print("PROCESSING: Fraud_Data.csv")
    print("=" * 60)

    df = load_fraud_data()
    print(f"\nRaw shape: {df.shape}")

    # Parse datetimes
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])

    # Clean
    df = handle_cleaning(df)
    print(f"After cleaning: {df.shape}")

    # Geolocation
    ip_df = load_ip_to_country()
    ip_ranges = prepare_ip_ranges(ip_df)
    df = add_ip_country(df, ip_ranges)
    df, significant_countries = add_country_risk_features(df)
    fraud_by_country = get_fraud_by_country(df)
    print(f"Countries mapped. Unique: {df['country'].nunique()}")

    # Temporal features
    df = engineer_temporal_features(df)

    # Velocity features
    df = engineer_velocity_features(df)

    # Encode categoricals
    categorical_cols = ["source", "browser", "sex", "time_of_day"]
    df = encode_categoricals(df, categorical_cols)

    # Save full processed frame
    _ensure_dir(DATA_PROCESSED)
    df.to_csv(DATA_PROCESSED / "fraud_data_features.csv", index=False)
    print(f"Saved fraud_data_features.csv  shape={df.shape}")

    return df, fraud_by_country, significant_countries


def process_creditcard_data():
    print("=" * 60)
    print("PROCESSING: creditcard.csv")
    print("=" * 60)

    df = load_creditcard_data()
    print(f"\nRaw shape: {df.shape}")

    # Clean
    df = handle_cleaning(df)
    print(f"After cleaning: {df.shape}")

    # Feature engineering
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Time_hours"] = df["Time"] / 3600

    # Save
    _ensure_dir(DATA_PROCESSED)
    df.to_csv(DATA_PROCESSED / "creditcard_features.csv", index=False)
    print(f"Saved creditcard_features.csv  shape={df.shape}")

    return df


def prepare_modeling_data_fraud(
    df,
    target_col="class",
    test_size=0.2,
    smote_ratio=0.5,
):
    exclude_cols = [
        "user_id",
        "signup_time",
        "purchase_time",
        "device_id",
        "ip_address",
        "ip_address_int",
        "country",
        target_col,
        "signup_day",
        "purchase_day",
        "time_of_day",
        "hour_of_day",
        "day_of_week",
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    object_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(object_cols) > 0:
        print(f"Dropping non-numeric columns: {list(object_cols)}")
        X = X.drop(columns=object_cols)
        feature_cols = [c for c in feature_cols if c not in object_cols]

    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size)

    numerical_cols = [
        c
        for c in [
            "purchase_value",
            "age",
            "time_since_signup",
            "user_total_transactions",
            "device_total_transactions",
            "user_historical_fraud_rate",
            "device_historical_fraud_rate",
            "users_per_device",
            "devices_per_user",
            "signup_hour",
            "purchase_hour",
            "signup_month",
            "purchase_month",
        ]
        if c in X_train.columns
    ]
    X_train, X_test, scaler = scale_features(X_train, X_test, numerical_cols)

    X_train_smote, y_train_smote = apply_smote(X_train, y_train, smote_ratio)

    _ensure_dir(DATA_PROCESSED)
    for name, obj in [
        ("X_train_fraud", X_train),
        ("X_test_fraud", X_test),
        ("y_train_fraud", y_train),
        ("y_test_fraud", y_test),
        (
            "X_train_fraud_smote",
            pd.DataFrame(X_train_smote, columns=feature_cols),
        ),
        (
            "y_train_fraud_smote",
            pd.Series(y_train_smote),
        ),
    ]:
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)
        else:
            obj.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"After SMOTE: {X_train_smote.shape}")
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_smote,
        y_train_smote,
        feature_cols,
    )


def prepare_modeling_data_credit(
    df, target_col="Class", test_size=0.2, smote_ratio=0.3
):
    v_cols = [f"V{i}" for i in range(1, 29)]
    feature_cols = v_cols + ["Amount_log", "Time_hours"]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size)

    X_train, X_test, scaler = scale_features(
        X_train, X_test, ["Amount_log", "Time_hours"]
    )

    X_train_smote, y_train_smote = apply_smote(X_train, y_train, smote_ratio)

    _ensure_dir(DATA_PROCESSED)
    for name, obj in [
        ("X_train_credit", X_train),
        ("X_test_credit", X_test),
        ("y_train_credit", y_train),
        ("y_test_credit", y_test),
        (
            "X_train_credit_smote",
            pd.DataFrame(X_train_smote, columns=feature_cols),
        ),
        (
            "y_train_credit_smote",
            pd.Series(y_train_smote),
        ),
    ]:
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)
        else:
            obj.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"After SMOTE: {X_train_smote.shape}")
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_smote,
        y_train_smote,
        feature_cols,
    )
