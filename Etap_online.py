import numpy as np
import pandas as pd
import time

from scipy.stats import t

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from river import tree, compose, preprocessing



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

K_CBR = 3
CBR_RETAIN = True

MIN_SAMPLES_FOR_LOCAL_MODEL = 50


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", None)




# 2. METRYKI
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



# 3. PRZYGOTOWANIE DANYCH
def add_time_features(df):
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

    df = df.dropna(subset=required_columns).copy()
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

    df = df.reset_index(drop=True)

    print("Użyte cechy:")
    for feature in features:
        print(f"- {feature}")

    return df, features




# 4. BASELINE
def predict_group_median_baseline(train_df, test_df):
    global_median = train_df[TARGET].median()
    group_median = train_df.groupby("model_group")[TARGET].median()

    y_pred = (
        test_df["model_group"]
        .map(group_median)
        .fillna(global_median)
        .values
    )

    return y_pred



# 5A. REGRESJA LINIOWA GLOBALNA Z PEŁNYM RETRENINGIEM
def predict_with_global_retrained_linear_regression(train_df, test_df, numeric_features):
    """
    regresja liniowa offline, która po każdym nowym rekordzie testowym
    jest ponownie trenowana na całym rosnącym zbiorze danych.
    """
    current_train_df = train_df.copy().reset_index(drop=True)

    model = LinearRegression()
    model.fit(current_train_df[numeric_features], current_train_df[TARGET])

    y_pred = []

    for _, row in test_df.iterrows():
        X_row = row[numeric_features].to_frame().T

        pred = model.predict(X_row)[0]
        y_pred.append(float(pred))

        # po poznaniu rzeczywistego duration_seconds rekord zostaje dodany
        # do bazy wiedzy, a model jest trenowany od nowa na całym zbiorze
        current_train_df = pd.concat(
            [current_train_df, row.to_frame().T],
            ignore_index=True
        )

        model = LinearRegression()
        model.fit(current_train_df[numeric_features], current_train_df[TARGET])

    return y_pred


# 5B. REGRESJA LINIOWA GLOBAL/LOCAL Z DOUCZANIEM
def train_linear_models(train_df, numeric_features):
    global_model = LinearRegression()
    global_model.fit(train_df[numeric_features], train_df[TARGET])

    local_models = {}

    for model_group, group in train_df.groupby("model_group"):
        if len(group) >= MIN_SAMPLES_FOR_LOCAL_MODEL:
            local_model = LinearRegression()
            local_model.fit(group[numeric_features], group[TARGET])
            local_models[model_group] = local_model

    return global_model, local_models


def predict_with_updated_linear_regression(train_df, test_df, numeric_features):
    current_train_df = train_df.copy().reset_index(drop=True)

    global_model, local_models = train_linear_models(
        current_train_df,
        numeric_features
    )

    y_pred = []

    for _, row in test_df.iterrows():
        model_group = row["model_group"]
        X_row = row[numeric_features].to_frame().T

        if model_group in local_models:
            pred = local_models[model_group].predict(X_row)[0]
        else:
            pred = global_model.predict(X_row)[0]

        y_pred.append(float(pred))

        # douczenie po poznaniu prawdziwego duration_seconds
        current_train_df = pd.concat(
            [current_train_df, row.to_frame().T],
            ignore_index=True
        )

        global_model, local_models = train_linear_models(
            current_train_df,
            numeric_features
        )

    return y_pred



# 6. HOEFFDING TREE
def row_to_river_features(row, numeric_features):
    x = {}

    for feature in numeric_features:
        x[feature] = float(row[feature])

    x["model_group"] = str(row["model_group"])

    return x


def predict_with_hoeffding_tree(train_df, test_df, numeric_features):
    model = (
        (compose.Select("model_group") | preprocessing.OneHotEncoder())
        + compose.Select(*numeric_features)
        | tree.HoeffdingTreeRegressor(
            grace_period=1,
            delta=1,
            max_depth=30
        )
    )

    for _, row in train_df.iterrows():
        x = row_to_river_features(row, numeric_features)
        y = float(row[TARGET])
        model.learn_one(x, y)

    global_median = train_df[TARGET].median()

    y_pred = []

    for _, row in test_df.iterrows():
        x = row_to_river_features(row, numeric_features)
        y = float(row[TARGET])

        pred = model.predict_one(x)

        if pred is None or pd.isna(pred):
            pred = global_median

        y_pred.append(float(pred))

        # douczenie po poznaniu prawdziwego duration_seconds
        model.learn_one(x, y)

    return y_pred



# 7. CBR
def normalize_train_test(train_df, test_df, numeric_features):
    train_norm = train_df.copy()
    test_norm = test_df.copy()

    for feature in numeric_features:
        min_value = train_norm[feature].min()
        max_value = train_norm[feature].max()

        denominator = max_value - min_value

        norm_col = feature + "_normalized"

        if denominator == 0:
            train_norm[norm_col] = 0.0
            test_norm[norm_col] = 0.0
        else:
            train_norm[norm_col] = (train_norm[feature] - min_value) / denominator
            test_norm[norm_col] = (test_norm[feature] - min_value) / denominator

    return train_norm, test_norm


def cbr_predict_one(train_cbr, row, normalized_features, k):
    same_group = train_cbr[train_cbr["model_group"] == row["model_group"]]

    if len(same_group) >= k:
        candidates = same_group
    else:
        candidates = train_cbr

    row_values = row[normalized_features].astype(float).values
    candidate_values = candidates[normalized_features].astype(float).values

    #odległość euklidesowa
    #distances = np.sqrt(((candidate_values - row_values) ** 2).sum(axis=1))

    #odległość manhattan
    distances = np.abs(candidate_values - row_values).sum(axis=1)

    nearest_indices = np.argsort(distances)[:k]
    nearest_durations = candidates.iloc[nearest_indices][TARGET].astype(float)

    return nearest_durations.mean()


def predict_with_cbr(train_df, test_df, numeric_features):
    train_cbr, test_cbr = normalize_train_test(
        train_df,
        test_df,
        numeric_features
    )

    normalized_features = [
        feature + "_normalized"
        for feature in numeric_features
    ]

    y_pred = []

    for _, row in test_cbr.iterrows():
        pred = cbr_predict_one(
            train_cbr=train_cbr,
            row=row,
            normalized_features=normalized_features,
            k=K_CBR
        )

        y_pred.append(float(pred))

        if CBR_RETAIN:
            train_cbr = pd.concat(
                [train_cbr, row.to_frame().T],
                ignore_index=True
            )

    return y_pred




# 8. PODSUMOWANIE WYNIKÓW
def summarize_results(results_df):
    summary_rows = []

    metrics = ["MAE", "RMSE", "SMAPE", "R2", "Fold_time_seconds"]

    for model in results_df["Model"].unique():
        model_results = results_df[results_df["Model"] == model]

        row = {
            "Model": model
        }

        n = len(model_results)
        t_value = t.ppf(0.975, df=n - 1)

        for metric in metrics:
            values = model_results[metric].values

            mean_value = np.mean(values)
            std_value = np.std(values, ddof=1)
            ci95 = t_value * std_value / np.sqrt(n)

            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_95CI"] = ci95
            row[f"{metric}_mean±CI"] = f"{mean_value:.4f} ± {ci95:.4f}"

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("MAE_mean")

    return summary_df



# 9. K-FOLD DLA MODELI ONLINE
def run_model_on_fold(model_name, predict_function, train_df, test_df, y_true, numeric_features):
    """
    Uruchamia pojedynczy model dla jednego folda i mierzy cały czas pracy modelu
    Dla modeli online/adaptacyjnych obejmuje to:
    -przygotowanie/trening początkowy na train_df,
    -predykcję rekordów z test_df,
    -aktualizację wiedzy po każdym rekordzie testowym.
    """
    start_time = time.perf_counter()

    y_pred = predict_function(
        train_df=train_df,
        test_df=test_df,
        numeric_features=numeric_features
    )

    elapsed_time = time.perf_counter() - start_time

    metrics = calculate_metrics(y_true, y_pred)
    metrics["Fold_time_seconds"] = elapsed_time

    print(f"{model_name} done | czas folda: {elapsed_time:.4f} s")

    return metrics


def run_kfold_experiment(df, features):
    print("\n=== START K-FOLD CROSS VALIDATION - ONLINE ===")

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    numeric_features = [
        feature
        for feature in features
        if feature != "model_group"
    ]

    all_results = []

    for fold_number, (train_index, test_index) in enumerate(kfold.split(df), start=1):
        print(f"\n--- Fold {fold_number}/{N_SPLITS} ---")

        train_df = df.iloc[train_index].copy().reset_index(drop=True)
        test_df = df.iloc[test_index].copy().reset_index(drop=True)

        y_true = test_df[TARGET].astype(float).values


        # BASELINE GROUP MEDIAN
        start_time = time.perf_counter()

        y_pred_baseline = predict_group_median_baseline(train_df, test_df)

        elapsed_time = time.perf_counter() - start_time

        metrics_baseline = calculate_metrics(y_true, y_pred_baseline)
        metrics_baseline["Fold_time_seconds"] = elapsed_time

        all_results.append({
            "Fold": fold_number,
            "Model": "Baseline group median",
            **metrics_baseline
        })

        print(f"Baseline group median done | czas folda: {elapsed_time:.4f} s")


        # REGRESJA LINIOWA GLOBALNA Z PEŁNYM RETRENINGIEM
        metrics_lr_global = run_model_on_fold(
            model_name="Regresja liniowa global retraining",
            predict_function=predict_with_global_retrained_linear_regression,
            train_df=train_df,
            test_df=test_df,
            y_true=y_true,
            numeric_features=numeric_features
        )

        all_results.append({
            "Fold": fold_number,
            "Model": "Regresja liniowa global retraining",
            **metrics_lr_global
        })


        # REGRESJA LINIOWA GLOBAL/LOCAL Z DOUCZANIEM
        metrics_lr_global_local = run_model_on_fold(
            model_name="Regresja liniowa global/local",
            predict_function=predict_with_updated_linear_regression,
            train_df=train_df,
            test_df=test_df,
            y_true=y_true,
            numeric_features=numeric_features
        )

        all_results.append({
            "Fold": fold_number,
            "Model": "Regresja liniowa global/local",
            **metrics_lr_global_local
        })


        # HOEFFDING TREE
        metrics_ht = run_model_on_fold(
            model_name="Hoeffding Tree",
            predict_function=predict_with_hoeffding_tree,
            train_df=train_df,
            test_df=test_df,
            y_true=y_true,
            numeric_features=numeric_features
        )

        all_results.append({
            "Fold": fold_number,
            "Model": "Hoeffding Tree",
            **metrics_ht
        })


        # CBR
        metrics_cbr = run_model_on_fold(
            model_name=f"CBR retain={CBR_RETAIN}, k={K_CBR}",
            predict_function=predict_with_cbr,
            train_df=train_df,
            test_df=test_df,
            y_true=y_true,
            numeric_features=numeric_features
        )

        all_results.append({
            "Fold": fold_number,
            "Model": f"CBR retain={CBR_RETAIN}, k={K_CBR}",
            **metrics_cbr
        })

    return pd.DataFrame(all_results)



# 10. URUCHOMIENIE
def main():
    print("============================================")
    print("ETAP ONLINE - K-FOLD CROSS VALIDATION")
    print("============================================")

    print(f"CSV_PATH = {CSV_PATH}")
    print(f"USE_1_99 = {USE_1_99}")
    print(f"USE_TIME_FEATURES = {USE_TIME_FEATURES}")
    print(f"N_SPLITS = {N_SPLITS}")
    print(f"K_CBR = {K_CBR}")
    print(f"CBR_RETAIN = {CBR_RETAIN}")
    print(f"MIN_SAMPLES_FOR_LOCAL_MODEL = {MIN_SAMPLES_FOR_LOCAL_MODEL}")
    print("Czas folda obejmuje trening początkowy, predykcje i aktualizacje modelu w obrębie folda.")

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