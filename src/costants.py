# Column data types for Fraud_Data.csv
FRAUD_COLUMN_DATA_TYPES = {
    "user_id": "int64",
    "signup_time": "datetime64[ns]",
    "purchase_time": "datetime64[ns]",
    "purchase_value": "float64",
    "device_id": "str",
    "source": "str",
    "browser": "str",
    "sex": "str",
    "age": "int64",
    "ip_address": "float64",
    "class": "int64",
}

# Column data types for creditcard.csv
CREDIT_CARD_COLUMN_DATA_TYPES = {
    "Time": "float64",
    "V1": "float64",
    "V2": "float64",
    "V3": "float64",
    "V4": "float64",
    "V5": "float64",
    "V6": "float64",
    "V7": "float64",
    "V8": "float64",
    "V9": "float64",
    "V10": "float64",
    "V11": "float64",
    "V12": "float64",
    "V13": "float64",
    "V14": "float64",
    "V15": "float64",
    "V16": "float64",
    "V17": "float64",
    "V18": "float64",
    "V19": "float64",
    "V20": "float64",
    "V21": "float64",
    "V22": "float64",
    "V23": "float64",
    "V24": "float64",
    "V25": "float64",
    "V26": "float64",
    "V27": "float64",
    "V28": "float64",
    "Amount": "float64",
    "Class": "int64",
}

# Column data types for IpAddress_to_Country.csv
IP_ADDRESS_COLUMN_DATA_TYPES = {
    "lower_bound_ip_address": "int64",
    "upper_bound_ip_address": "int64",
    "country": "str",
}

# Feature columns for modeling
FRAUD_FEATURE_COLUMNS = [
    "purchase_value",
    "age",
    "signup_hour",
    "purchase_hour",
    "signup_month",
    "purchase_month",
    "signup_is_weekend",
    "purchase_is_weekend",
    "time_since_signup",
    "user_total_transactions",
    "device_total_transactions",
    "user_historical_fraud_rate",
    "device_historical_fraud_rate",
    "users_per_device",
    "devices_per_user",
    "is_high_risk_country",
    "is_unknown_country",
    "source_Direct",
    "source_SEO",
    "browser_Firefox",
    "browser_IE",
    "browser_Opera",
    "browser_Safari",
    "sex_M",
    "time_of_day_evening",
    "time_of_day_morning",
    "time_of_day_night",
]

CREDIT_FEATURE_COLUMNS = [
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount_log",
    "Time_hours",
]

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# File paths
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

# Dataset file names
FRAUD_DATA_FILE = "Fraud_Data.csv"
CREDIT_CARD_DATA_FILE = "creditcard.csv"
IP_ADDRESS_DATA_FILE = "IpAddress_to_Country.csv"
