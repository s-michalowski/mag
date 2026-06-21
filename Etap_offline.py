import numpy as np
import pandas as pd
import time

from scipy.stats import t

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# 1. KONFIGURACJA EKSPERYMENTU
CSV_PATH = "wynik.csv"

USE_1_99 = True
USE_TIME_FEATURES = True

N_SPLITS = 5
RANDOM_STATE = 42

TARGET = "duration_seconds"

BASE_FEATURES = [
    "pixel_count",
    "compression_ratio",
    "complexity",
    "model_group"
]

TIME_FEATURES = [
    "start_hour",
    "start_dayofweek"
]



# 2. FUNKCJE POMOCNICZE
def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "SMAPE": smape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


def add_time_features(df):
    """
    Dodaje cechy czasowe:
    - start_hour,
    - start_dayofweek.
    """
    df = df.copy()

    df["startedAtDate"] = pd.to_datetime(
        df["startedAtDate"],
        errors="coerce",
        dayfirst=True
    )

    df["start_hour"] = df["startedAtDate"].dt.hour
    df["start_dayofweek"] = df["startedAtDate"].dt.dayofweek

    return df


def prepare_data():
    """
    Wczytuje dane, filtruje status NEW, usuwa braki i opcjonalnie:
    - dodaje cechy czasowe,
    - filtruje zakres 1–99%.
    """
    print("=== WCZYTYWANIE DANYCH ===")

    df = pd.read_csv(CSV_PATH)

    print(f"Liczba rekordów przed filtrowaniem: {len(df)}")

    df = df[df["status"] == "NEW"].copy()

    print(f"Liczba rekordów po status == NEW: {len(df)}")

    features = BASE_FEATURES.copy()

    if USE_TIME_FEATURES:
        print("Dodawanie cech czasowych...")
        df = add_time_features(df)
        features += TIME_FEATURES

    required_columns = features + [TARGET]

    df = df.dropna(subset=required_columns)
    df = df[df[TARGET] > 0].copy()

    print(f"Liczba rekordów po usunięciu braków i duration <= 0: {len(df)}")

    if USE_1_99:
        q01 = df[TARGET].quantile(0.01)
        q99 = df[TARGET].quantile(0.99)

        df = df[(df[TARGET] >= q01) & (df[TARGET] <= q99)].copy()

        print("Zastosowano filtrowanie 1–99%.")
        print(f"q01 = {q01:.4f}")
        print(f"q99 = {q99:.4f}")
        print(f"Liczba rekordów po 1–99%: {len(df)}")

    print("Użyte cechy:")
    for feature in features:
        print(f"- {feature}")

    return df, features


def encode_features(X_train, X_test):
    """
    One-hot encoding dla model_group.
    Kodowanie wykonywane osobno wewnątrz każdego foldu,
    żeby nie mieszać zbioru treningowego i testowego.
    """
    X_train_encoded = pd.get_dummies(X_train, columns=["model_group"], drop_first=False)
    X_test_encoded = pd.get_dummies(X_test, columns=["model_group"], drop_first=False)

    X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded,
        join="left",
        axis=1,
        fill_value=0
    )

    return X_train_encoded, X_test_encoded


def predict_baselines(train_df, test_df):
    """
        1. Globalna mediana jakby model_group nie był wystarczająco reprezentatywny
        2. Mediana dla model_group.
    """
    y_train = train_df[TARGET]

    global_median = y_train.median()
    group_median = train_df.groupby("model_group")[TARGET].median()

    predictions = {
        "Baseline group median": test_df["model_group"].map(group_median).fillna(global_median).values
    }

    return predictions


def summarize_results(results_df):
    """
    Tworzy tabelę:
    - średnia,
    - odchylenie standardowe,
    - 95% confidence interval,
    - zapis: mean ± CI.
    """
    summary_rows = []

    models = results_df["Model"].unique()
    metrics = ["MAE", "RMSE", "SMAPE", "R2", "Fold_time_seconds"]

    for model in models:
        model_results = results_df[results_df["Model"] == model]

        row = {
            "Model": model
        }

        for metric in metrics:
            values = model_results[metric].values

            mean_value = np.mean(values)
            std_value = np.std(values, ddof=1)

            n = len(values)
            t_value = t.ppf(0.975, df=n - 1)
            ci95 = t_value * std_value / np.sqrt(n)

            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_95CI"] = ci95
            row[f"{metric}_mean±CI"] = f"{mean_value:.4f} ± {ci95:.4f}"
            row[f"{metric}_mean±std"] = f"{mean_value:.4f} ± {std_value:.4f}"

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)



# 3.  K-FOLD
def run_kfold_experiment(df, features):
    """
    Wykonuje K-Fold Cross Validation dla:
    - baseline'u,
    - Linear Regression,
    - Random Forest,
    - Neural Network.
    """
    print("\n=== START K-FOLD CROSS VALIDATION ===")

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    all_results = []

    for fold_number, (train_index, test_index) in enumerate(kfold.split(df), start=1):
        print(f"\n--- Fold {fold_number}/{N_SPLITS} ---")

        train_df = df.iloc[train_index].copy()
        test_df = df.iloc[test_index].copy()

        X_train = train_df[features]
        y_train = train_df[TARGET]

        X_test = test_df[features]
        y_test = test_df[TARGET]

        X_train_encoded, X_test_encoded = encode_features(X_train, X_test)

        # BASELINE
        start_time = time.perf_counter()

        baseline_predictions = predict_baselines(train_df, test_df)

        for baseline_name, y_pred in baseline_predictions.items():
            fold_time = time.perf_counter() - start_time
            metrics = calculate_metrics(y_test, y_pred)

            all_results.append({
                "Fold": fold_number,
                "Model": baseline_name,
                "Fold_time_seconds": fold_time,
                **metrics
            })

        print("Baseline group median done")

        # LINEAR REGRESSION
        start_time = time.perf_counter()

        linear_model = LinearRegression()
        linear_model.fit(X_train_encoded, y_train)

        y_pred_linear = linear_model.predict(X_test_encoded)
        fold_time = time.perf_counter() - start_time

        metrics_linear = calculate_metrics(y_test, y_pred_linear)

        all_results.append({
            "Fold": fold_number,
            "Model": "Linear Regression",
            "Fold_time_seconds": fold_time,
            **metrics_linear
        })

        print("Linear Regression done")

        # RANDOM FOREST
        start_time = time.perf_counter()

        random_forest = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        random_forest.fit(X_train_encoded, y_train)

        y_pred_rf = random_forest.predict(X_test_encoded)
        fold_time = time.perf_counter() - start_time

        metrics_rf = calculate_metrics(y_test, y_pred_rf)

        all_results.append({
            "Fold": fold_number,
            "Model": "Random Forest",
            "Fold_time_seconds": fold_time,
            **metrics_rf
        })

        print("Random Forest done")


        # NEURAL NETWORK
        start_time = time.perf_counter()

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train_encoded)
        X_test_scaled = scaler.transform(X_test_encoded)

        neural_network = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=RANDOM_STATE
        )

        neural_network.fit(X_train_scaled, y_train)

        y_pred_nn = neural_network.predict(X_test_scaled)
        fold_time = time.perf_counter() - start_time

        metrics_nn = calculate_metrics(y_test, y_pred_nn)

        all_results.append({
            "Fold": fold_number,
            "Model": "Neural Network",
            "Fold_time_seconds": fold_time,
            **metrics_nn
        })

        print("Neural Network done")

    return pd.DataFrame(all_results)







# 4. URUCHOMIENIE
def main():
    print("============================================")
    print("ETAP OFFLINE - K-FOLD CROSS VALIDATION")
    print("============================================")

    print(f"CSV_PATH = {CSV_PATH}")
    print(f"USE_1_99 = {USE_1_99}")
    print(f"USE_TIME_FEATURES = {USE_TIME_FEATURES}")
    print(f"N_SPLITS = {N_SPLITS}")

    df, features = prepare_data()

    results_df = run_kfold_experiment(df, features)
    summary_df = summarize_results(results_df)

    print("\n=== WYNIKI SZCZEGÓŁOWE Z FOLDÓW ===")
    print(results_df.to_string(index=False))

    print("\n=== PODSUMOWANIE: MEAN, STD, 95% CI ===")

    summary_to_print = summary_df[
        [
            "Model",
            "MAE_mean±CI",
            "RMSE_mean±CI",
            "SMAPE_mean±CI",
            "R2_mean±CI",
            "Fold_time_seconds_mean±CI"
        ]
    ]

    print(summary_to_print.to_string(index=False))

    print("\n=== KONIEC EKSPERYMENTU ===")

if __name__ == "__main__":
    main()