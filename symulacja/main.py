import pandas as pd
import time

# OFFLINE
from modele_offline import (
    train_linear_regression,
    train_random_forest,
    train_neural_network,
    predict_single, train_baseline_group_median
)

# ONLINE
from modele_online import (
    OnlineLinearRegressionModel,
    HoeffdingTreeModel,
    CBRModel
)

# SYMULACJA
from simulation import simulate, print_simulation_result, save_simulation_details


TRAIN_DATA_PATH = "wynik.csv"     # pełne dane historyczne do trenowania
QUEUE_DATA_PATH = "kolejka.xlsx"    # kolejka do symulacji

MODEL_TYPE = "hoeffdingM"
# offline:
# "linear"
# "rf"
# "nn"
# "baseline"
#
# online:
# "online_lr"
# "hoeffding"
# "cbr"

MODE = "online"
# "offline" albo "online"

K = 3   # liczba workerów

# WCZYTYWANIE DANYCH
def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    #konwersja dat
    if "startedAtDate" in df.columns:
        df["startedAtDate"] = pd.to_datetime(df["startedAtDate"], errors="coerce")
    if "finishedAtDate" in df.columns:
        df["finishedAtDate"] = pd.to_datetime(df["finishedAtDate"], errors="coerce")

    #wybrane kolumny
    required_cols = [
        "pixel_count",
        "compression_ratio",
        "complexity",
        "model_group",
        "duration_seconds"
    ]

    #usunięcie rekordów pustych lub takich w których duration jest mniejsze od zerae
    df = df.dropna(subset=required_cols).copy()
    df = df[df["duration_seconds"] > 0].copy()

    # uczymy modele na rekordach o statusie NEW
    if "status" in df.columns:
        df = df[df["status"] == "NEW"].copy()

    return df.reset_index(drop=True)

#pobranie danych kolejki z kolejka.xlsx
def load_queue_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    df["startedAtDate"] = pd.to_datetime(df["startedAtDate"], errors="coerce")
    df["finishedAtDate"] = pd.to_datetime(df["finishedAtDate"], errors="coerce")

    #wybrane kolumny
    numeric_cols = [
        "pixel_count",
        "compression_ratio",
        "complexity",
        "duration_seconds"
    ]

    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = [
        "pixel_count",
        "compression_ratio",
        "complexity",
        "model_group",
        "duration_seconds",
        "startedAtDate",
        "finishedAtDate"
    ]

    df = df.dropna(subset=required_cols).copy()
    df = df[df["duration_seconds"] > 0].copy()

    return df.reset_index(drop=True)





# TRENING MODELI OFFLINE
def train_offline_model(model_type: str, train_df: pd.DataFrame):
    if model_type == "baseline":
        model, feature_cols = train_baseline_group_median(train_df)
        return model, feature_cols, None

    if model_type == "linear":
        model, feature_cols = train_linear_regression(train_df)
        return model, feature_cols, None

    if model_type == "rf":
        model, feature_cols = train_random_forest(train_df)
        return model, feature_cols, None

    if model_type == "nn":
        model, feature_cols, scaler = train_neural_network(train_df)
        return model, feature_cols, scaler

    raise ValueError("Nieznany model offline")


# TRENING MODELI ONLINE
def train_online_model(model_type: str, train_df: pd.DataFrame):
    if model_type == "online_lr":
        return OnlineLinearRegressionModel(train_df)

    if model_type == "hoeffding":
        return HoeffdingTreeModel(train_df)

    if model_type == "cbr":
        return CBRModel(train_df, k=3, retain=True)

    raise ValueError("Nieznany model online")




def main():
    print("=== START ===")

    # 1. wczytanie danych
    train_df = load_training_data(TRAIN_DATA_PATH)
    queue_df = load_queue_data(QUEUE_DATA_PATH)

    print(f"Liczba rekordów treningowych: {len(train_df)}")
    print(f"Liczba rekordów w kolejce: {len(queue_df)}")
    print(f"Liczba workerów k: {K}")
    print(f"Tryb: {MODE}")
    print(f"Model: {MODEL_TYPE}")

    start_time = time.perf_counter()

    # 2. trening / inicjalizacja modelu
    if MODE == "offline":
        model, feature_cols, scaler = train_offline_model(MODEL_TYPE, train_df)

        class OfflineWrapper:
            def __init__(self, trained_model, feature_cols_, scaler_):
                self.trained_model = trained_model
                self.feature_cols = feature_cols_
                self.scaler = scaler_

            def predict(self, row):
                return predict_single(
                    self.trained_model,
                    row,
                    self.feature_cols,
                    self.scaler
                )

        wrapped_model = OfflineWrapper(model, feature_cols, scaler)

        result = simulate(
            model=wrapped_model,
            df=queue_df,
            k=K,
            mode="offline",
            feature_cols=feature_cols
        )
    elif MODE == "online":
        model = train_online_model(MODEL_TYPE, train_df)

        result = simulate(
            model=model,
            df=queue_df,
            k=K,
            mode="online"
        )

    else:
        raise ValueError("MODEL musi być 'offline' albo 'online'")

    end_time = time.perf_counter()
    execution_time_seconds = end_time - start_time

    # 3. wynik
    print_simulation_result(result)

    print(f"Czas trenowania/inicjalizacji modelu + symulacji [s]: {execution_time_seconds:.6f}")

    output_file = f"simulation_details_{MODE}_{MODEL_TYPE}_k{K}.csv"
    save_simulation_details(result, output_file)
    print(f"Szczegóły symulacji zapisane do pliku: {output_file}")


if __name__ == "__main__":
    main()