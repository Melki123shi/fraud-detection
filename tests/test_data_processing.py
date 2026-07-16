import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.data_processing import (
    _ensure_dir,
    load_data,
    check_missing_values,
    check_duplicates,
    check_outliers,
    correct_data_types,
    handle_cleaning,
    save_cleaned_data,
    map_ip_to_country,
    prepare_ip_ranges,
    add_ip_country,
    add_country_risk_features,
    get_fraud_by_country,
    categorize_hour,
    engineer_temporal_features,
    engineer_velocity_features,
    get_class_distribution,
    get_fraud_rate_by_column,
    encode_categoricals,
    scale_features,
    stratified_split,
    apply_smote,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "device_id": ["a", "a", "b", "b", "c"],
        "ip_address": [
            16777216, 16777217, 3232235777, 3232235778, 67108864
        ],
        "signup_time": pd.to_datetime([
            "2024-01-01 10:00:00",
            "2024-01-01 10:00:00",
            "2024-02-01 14:00:00",
            "2024-02-01 14:00:00",
            "2024-03-01 02:00:00",
        ]),
        "purchase_time": pd.to_datetime([
            "2024-01-01 12:00:00",
            "2024-01-02 20:00:00",
            "2024-02-01 16:00:00",
            "2024-02-02 08:00:00",
            "2024-03-01 04:00:00",
        ]),
        "purchase_value": [50.0, 120.0, 30.0, 200.0, 15.0],
        "age": [25, 25, 30, 30, 22],
        "class": [0, 1, 0, 0, 1],
        "source": ["SEO", "Ads", "Direct", "SEO", "Ads"],
        "browser": ["Chrome", "Firefox", "Safari", "Chrome", "IE"],
        "sex": ["M", "F", "M", "F", "M"],
    })


@pytest.fixture
def ip_ranges_df():
    return pd.DataFrame({
        "lower_bound_ip_address": [16777216, 3232235520],
        "upper_bound_ip_address": [16777471, 3232235777],
        "country": ["United States", "China"],
    })


# ---------------------------------------------------------------------------
# _ensure_dir
# ---------------------------------------------------------------------------
class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "sub" / "dir"
        _ensure_dir(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_existing_directory(self, tmp_path):
        _ensure_dir(tmp_path)
        assert tmp_path.exists()


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------
class TestLoadData:
    def test_load_valid_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        df = load_data(csv_file)
        assert df is not None
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_load_nonexistent_file(self):
        result = load_data(Path("/nonexistent/file.csv"))
        assert result is None


# ---------------------------------------------------------------------------
# check_missing_values
# ---------------------------------------------------------------------------
class TestCheckMissingValues:
    def test_no_missing(self, sample_df):
        result = check_missing_values(sample_df)
        assert result.sum() == 0

    def test_with_missing(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        result = check_missing_values(df)
        assert result["a"] == 1
        assert result["b"] == 1

    def test_none_input(self):
        result = check_missing_values(None)
        assert result is None


# ---------------------------------------------------------------------------
# check_duplicates
# ---------------------------------------------------------------------------
class TestCheckDuplicates:
    def test_no_duplicates(self, sample_df):
        result = check_duplicates(sample_df)
        assert len(result) == 0

    def test_with_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = check_duplicates(df)
        assert len(result) == 1

    def test_none_input(self):
        result = check_duplicates(None)
        assert result is None


# ---------------------------------------------------------------------------
# check_outliers
# ---------------------------------------------------------------------------
class TestCheckOutliers:
    def test_no_outliers(self):
        df = pd.DataFrame({"val": [10, 11, 12, 10, 11]})
        result = check_outliers(df, "val")
        assert len(result) == 0

    def test_with_outliers(self):
        df = pd.DataFrame({"val": [10, 10, 10, 10, 100]})
        result = check_outliers(df, "val")
        assert len(result) == 1
        assert result["val"].iloc[0] == 100

    def test_missing_column(self, sample_df):
        result = check_outliers(sample_df, "nonexistent")
        assert result is None

    def test_none_input(self):
        result = check_outliers(None, "col")
        assert result is None


# ---------------------------------------------------------------------------
# correct_data_types
# ---------------------------------------------------------------------------
class TestCorrectDataTypes:
    def test_cast_types(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["1.1", "2.2", "3.3"],
        })
        result = correct_data_types(df, {"b": "float64"})
        assert result["b"].dtype == np.float64

    def test_nonexistent_column(self, sample_df):
        result = correct_data_types(
            sample_df, {"nonexistent": "float64"}
        )
        assert result is not None

    def test_none_input(self):
        result = correct_data_types(None, {"a": "int"})
        assert result is None


# ---------------------------------------------------------------------------
# handle_cleaning
# ---------------------------------------------------------------------------
class TestHandleCleaning:
    def test_cleans_data(self):
        df = pd.DataFrame({
            "a": [1, 1, np.nan, 4],
            "b": [np.nan, 2, 3, 4],
        })
        result = handle_cleaning(df)
        assert result is not None
        assert len(result) == 4
        assert result.isnull().sum().sum() == 0

    def test_none_input(self):
        result = handle_cleaning(None)
        assert result is None


# ---------------------------------------------------------------------------
# save_cleaned_data
# ---------------------------------------------------------------------------
class TestSaveCleanedData:
    def test_saves_csv(self, tmp_path, sample_df):
        file_path = tmp_path / "output.csv"
        save_cleaned_data(sample_df, file_path)
        assert file_path.exists()
        loaded = pd.read_csv(file_path)
        assert len(loaded) == len(sample_df)

    def test_none_input(self, tmp_path):
        file_path = tmp_path / "output.csv"
        save_cleaned_data(None, file_path)
        assert not file_path.exists()


# ---------------------------------------------------------------------------
# map_ip_to_country
# ---------------------------------------------------------------------------
class TestMapIpToCountry:
    def test_match_in_range(self, ip_ranges_df):
        result = map_ip_to_country(16777300, ip_ranges_df)
        assert result == "United States"

    def test_exact_lower_bound(self, ip_ranges_df):
        result = map_ip_to_country(16777216, ip_ranges_df)
        assert result == "United States"

    def test_exact_upper_bound(self, ip_ranges_df):
        result = map_ip_to_country(16777471, ip_ranges_df)
        assert result == "United States"

    def test_out_of_range(self, ip_ranges_df):
        result = map_ip_to_country(999999999, ip_ranges_df)
        assert result == "Unknown"

    def test_below_all_ranges(self, ip_ranges_df):
        result = map_ip_to_country(1, ip_ranges_df)
        assert result == "Unknown"


# ---------------------------------------------------------------------------
# prepare_ip_ranges
# ---------------------------------------------------------------------------
class TestPrepareIpRanges:
    def test_sorts_and_casts(self, ip_ranges_df):
        result = prepare_ip_ranges(ip_ranges_df)
        assert list(result.columns) == [
            "lower_bound_ip_address",
            "upper_bound_ip_address",
            "country",
        ]
        assert (
            result["lower_bound_ip_address"].is_monotonic_increasing
        )

    def test_types_are_int(self, ip_ranges_df):
        result = prepare_ip_ranges(ip_ranges_df)
        assert result["lower_bound_ip_address"].dtype == int
        assert result["upper_bound_ip_address"].dtype == int


# ---------------------------------------------------------------------------
# add_ip_country
# ---------------------------------------------------------------------------
class TestAddIpCountry:
    def test_adds_country_column(self, sample_df, ip_ranges_df):
        result = add_ip_country(sample_df, ip_ranges_df)
        assert "country" in result.columns
        assert "ip_address_int" in result.columns

    def test_country_values(self, sample_df, ip_ranges_df):
        result = add_ip_country(sample_df, ip_ranges_df)
        assert result["country"].iloc[0] == "United States"
        assert result["country"].iloc[2] == "China"


# ---------------------------------------------------------------------------
# add_country_risk_features
# ---------------------------------------------------------------------------
class TestAddCountryRiskFeatures:
    def test_adds_risk_columns(self, sample_df, ip_ranges_df):
        df = add_ip_country(sample_df, ip_ranges_df)
        result, stats = add_country_risk_features(df)
        assert "is_high_risk_country" in result.columns
        assert "is_unknown_country" in result.columns
        assert "fraud_rate" in stats.columns
        assert "total_transactions" in stats.columns

    def test_unknown_country_flag(self):
        df = pd.DataFrame({
            "country": ["Unknown", "US", "US", "US"],
            "class": [1, 0, 0, 1],
        })
        result, _ = add_country_risk_features(df, min_transactions=1)
        assert result["is_unknown_country"].iloc[0] == 1
        assert result["is_unknown_country"].iloc[1] == 0


# ---------------------------------------------------------------------------
# get_fraud_by_country
# ---------------------------------------------------------------------------
class TestGetFraudByCountry:
    def test_filters_and_sorts(self):
        df = pd.DataFrame({
            "country": ["A"] * 200 + ["B"] * 50 + ["C"] * 150,
            "class": [1] * 100 + [0] * 100
            + [1] * 40 + [0] * 10
            + [1] * 20 + [0] * 130,
        })
        result = get_fraud_by_country(df, min_transactions=100)
        assert "fraud_rate" in result.columns
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# categorize_hour
# ---------------------------------------------------------------------------
class TestCategorizeHour:
    def test_night(self):
        assert categorize_hour(0) == "night"
        assert categorize_hour(3) == "night"
        assert categorize_hour(5) == "night"

    def test_morning(self):
        assert categorize_hour(6) == "morning"
        assert categorize_hour(9) == "morning"
        assert categorize_hour(11) == "morning"

    def test_afternoon(self):
        assert categorize_hour(12) == "afternoon"
        assert categorize_hour(15) == "afternoon"
        assert categorize_hour(17) == "afternoon"

    def test_evening(self):
        assert categorize_hour(18) == "evening"
        assert categorize_hour(21) == "evening"
        assert categorize_hour(23) == "evening"


# ---------------------------------------------------------------------------
# engineer_temporal_features
# ---------------------------------------------------------------------------
class TestEngineerTemporalFeatures:
    def test_creates_all_columns(self, sample_df):
        result = engineer_temporal_features(sample_df)
        expected = [
            "signup_hour", "signup_day", "signup_month",
            "signup_is_weekend", "purchase_hour",
            "purchase_day", "purchase_month",
            "purchase_is_weekend", "time_since_signup",
            "hour_of_day", "day_of_week", "time_of_day",
        ]
        for col in expected:
            assert col in result.columns

    def test_time_since_signup_positive(self, sample_df):
        result = engineer_temporal_features(sample_df)
        assert (result["time_since_signup"] >= 0).all()

    def test_weekend_flag(self):
        df = pd.DataFrame({
            "signup_time": pd.to_datetime(
                ["2024-01-06 10:00:00"]
            ),
            "purchase_time": pd.to_datetime(
                ["2024-01-06 12:00:00"]
            ),
        })
        result = engineer_temporal_features(df)
        assert result["signup_is_weekend"].iloc[0] == 1

    def test_not_weekday_flag(self):
        df = pd.DataFrame({
            "signup_time": pd.to_datetime(
                ["2024-01-08 10:00:00"]
            ),
            "purchase_time": pd.to_datetime(
                ["2024-01-08 12:00:00"]
            ),
        })
        result = engineer_temporal_features(df)
        assert result["signup_is_weekend"].iloc[0] == 0


# ---------------------------------------------------------------------------
# engineer_velocity_features
# ---------------------------------------------------------------------------
class TestEngineerVelocityFeatures:
    def test_creates_velocity_columns(self, sample_df):
        result = engineer_velocity_features(sample_df)
        expected = [
            "user_total_transactions",
            "device_total_transactions",
            "user_historical_fraud_rate",
            "device_historical_fraud_rate",
            "users_per_device",
            "devices_per_user",
        ]
        for col in expected:
            assert col in result.columns

    def test_user_transaction_count(self, sample_df):
        result = engineer_velocity_features(sample_df)
        user1 = result[result["user_id"] == 1]
        assert (user1["user_total_transactions"] == 2).all()

    def test_user_fraud_rate(self, sample_df):
        result = engineer_velocity_features(sample_df)
        user3 = result[result["user_id"] == 3]
        assert user3["user_historical_fraud_rate"].iloc[0] == 1.0

    def test_rolling_windows(self, sample_df):
        result = engineer_velocity_features(sample_df)
        assert "user_transactions_24h" in result.columns
        assert "user_transactions_7d" in result.columns
        assert "user_transactions_30d" in result.columns

    def test_no_purchase_time_column(self):
        df = pd.DataFrame({
            "user_id": [1, 1],
            "device_id": ["a", "a"],
            "class": [0, 1],
        })
        result = engineer_velocity_features(df)
        assert "user_transactions_24h" not in result.columns


# ---------------------------------------------------------------------------
# get_class_distribution
# ---------------------------------------------------------------------------
class TestGetClassDistribution:
    def test_returns_correct_structure(self):
        y = np.array([0, 0, 0, 1, 1])
        result = get_class_distribution(y, name="test")
        assert 0 in result
        assert 1 in result
        assert result[0]["count"] == 3
        assert result[1]["count"] == 2

    def test_percentage_sums_to_100(self):
        y = np.array([0, 1, 2, 2])
        result = get_class_distribution(y)
        total_pct = sum(v["percentage"] for v in result.values())
        assert abs(total_pct - 100) < 0.01


# ---------------------------------------------------------------------------
# get_fraud_rate_by_column
# ---------------------------------------------------------------------------
class TestGetFraudRateByColumn:
    def test_returns_sorted(self, sample_df):
        result = get_fraud_rate_by_column(
            sample_df, "source", target="class"
        )
        assert "fraud_rate" in result.columns
        assert "total_transactions" in result.columns
        rates = result["fraud_rate"].values
        assert list(rates) == sorted(rates, reverse=True)

    def test_percentage_values(self):
        df = pd.DataFrame({
            "cat": ["A"] * 10 + ["B"] * 10,
            "class": [1] * 5 + [0] * 5 + [1] * 2 + [0] * 8,
        })
        result = get_fraud_rate_by_column(df, "cat")
        assert (result["fraud_rate"] <= 100).all()


# ---------------------------------------------------------------------------
# encode_categoricals
# ---------------------------------------------------------------------------
class TestEncodeCategoricals:
    def test_creates_dummies(self, sample_df):
        result = encode_categoricals(
            sample_df, ["source", "browser"]
        )
        assert "source_SEO" in result.columns
        assert "browser_Firefox" in result.columns

    def test_drop_first(self, sample_df):
        result = encode_categoricals(
            sample_df, ["source"], drop_first=True
        )
        assert "source_Ads" not in result.columns
        assert "source_SEO" in result.columns

    def test_no_drop(self, sample_df):
        result = encode_categoricals(
            sample_df, ["source"], drop_first=False
        )
        assert "source_Direct" in result.columns
        assert "source_SEO" in result.columns
        assert "source_Ads" in result.columns


# ---------------------------------------------------------------------------
# scale_features
# ---------------------------------------------------------------------------
class TestScaleFeatures:
    def test_scales_columns(self):
        X_train = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        })
        X_test = pd.DataFrame({
            "a": [6, 7],
            "b": [60, 70],
        })
        X_tr, X_te, scaler = scale_features(
            X_train, X_test, ["a", "b"]
        )
        assert abs(X_tr["a"].mean()) < 0.01
        var = X_tr["a"].var(ddof=0)
        assert abs(var - 1) < 0.01
        assert scaler is not None

    def test_does_not_modify_original(self):
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        X_test = pd.DataFrame({"a": [4.0, 5.0]})
        original_train = X_train["a"].copy()
        scale_features(X_train, X_test, ["a"])
        assert X_train["a"].equals(original_train)


# ---------------------------------------------------------------------------
# stratified_split
# ---------------------------------------------------------------------------
class TestStratifiedSplit:
    def test_split_sizes(self):
        X = pd.DataFrame({"a": range(100)})
        y = pd.Series([0] * 80 + [1] * 20)
        X_tr, X_te, y_tr, y_te = stratified_split(
            X, y, test_size=0.2
        )
        assert len(X_tr) == 80
        assert len(X_te) == 20

    def test_preserves_class_ratio(self):
        X = pd.DataFrame({"a": range(100)})
        y = pd.Series([0] * 80 + [1] * 20)
        X_tr, X_te, y_tr, y_te = stratified_split(
            X, y, test_size=0.2
        )
        train_ratio = y_tr.mean()
        test_ratio = y_te.mean()
        assert abs(train_ratio - test_ratio) < 0.05


# ---------------------------------------------------------------------------
# apply_smote
# ---------------------------------------------------------------------------
class TestApplySmote:
    def test_balances_minority(self):
        X = pd.DataFrame({"a": list(range(100)) * 1, "b": [0] * 100})
        y = pd.Series([0] * 90 + [1] * 10)
        X_res, y_res = apply_smote(
            X, y, sampling_strategy=0.5, random_state=42
        )
        assert len(y_res) > len(y)
        assert y_res.sum() > y.sum()

    def test_output_types(self):
        X = pd.DataFrame({"a": list(range(100)), "b": [0] * 100})
        y = pd.Series([0] * 90 + [1] * 10)
        X_res, y_res = apply_smote(
            X, y, sampling_strategy=0.5, random_state=42
        )
        assert isinstance(X_res, pd.DataFrame)
        assert isinstance(y_res, pd.Series)
