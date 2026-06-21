import numpy as np
import pandas as pd



#funkcje pomocnicze
def smape_single(y_pred: float, y_true: float) -> float:
    denominator = abs(y_pred) + abs(y_true)

    if denominator == 0:
        return 0.0
    return 100.0 * (2.0 * abs(y_pred - y_true) / denominator)

#zamiana date -> seconds (np: "2025-02-08 20:03:00" -> 1739044980.0)
def get_timestamp_seconds(ts) -> float:
    return pd.to_datetime(ts).timestamp()

#zamiana seconds -> date (np. 1639034980.0 -> "2021-12-09 07:29:40")
def seconds_to_datetime_str(seconds_value: float) -> str:
    return pd.to_datetime(seconds_value, unit="s").strftime("%Y-%m-%d %H:%M:%S")

#to samo co wyżej tylko że dla całej kolumny
def seconds_to_datetime_series(seconds_series: pd.Series) -> pd.Series:
    return pd.to_datetime(seconds_series, unit="s")



# chat wygenerował - pomaga w zapisaniu do plików csv, dzięki czemu możemy wykresy robić itd.
#trochę lepiej wykresy wyglądają te z excela niż te z matplotliba
def enrich_details_dataframe(details_df: pd.DataFrame) -> pd.DataFrame:
    df = details_df.copy()

    # błędy czasu wykonania pojedynczego zadania
    df["duration_error_seconds"] = (
        df["predicted_duration_seconds"] - df["real_duration_seconds"]
    )
    df["duration_abs_error_seconds"] = df["duration_error_seconds"].abs()

    #błąd SMAPE
    df["duration_smape"] = df.apply(
        lambda row: smape_single(
            row["predicted_duration_seconds"],
            row["real_duration_seconds"]
        ),
        axis=1
    )

    # błędy czasu zakończenia pojedynczego zadania
    df["end_time_error_seconds"] = (
        df["predicted_end_time"] - df["real_end_time"]
    )
    df["end_time_abs_error_seconds"] = df["end_time_error_seconds"].abs()

    df["end_time_smape"] = df.apply(
        lambda row: smape_single(
            row["predicted_end_time"],
            row["real_end_time"]
        ),
        axis=1
    )

    # czytelne kolumny datetime
    df["predicted_start_datetime"] = seconds_to_datetime_series(df["predicted_start_time"])
    df["predicted_end_datetime"] = seconds_to_datetime_series(df["predicted_end_time"])
    df["real_start_datetime"] = seconds_to_datetime_series(df["real_start_time"])
    df["real_end_datetime"] = seconds_to_datetime_series(df["real_end_time"])

    return df



# SYMULACJA OFFLINE
def simulate_queue_offline(model, df: pd.DataFrame, k: int):
    #skopiowanie danych wejściowych i zresetowanie indeksów (kolejność obrazów jest taka jak użytkownik podał)
    queue_df = df.copy().reset_index(drop=True)

    #bierzemy czas rozpoczęcia wykonywania się kolejki jako pierwszy czas z startedAtDate
    t0 = get_timestamp_seconds(queue_df.iloc[0]["startedAtDate"])
    #ustalenie ile mamy workerów i w którym czasie zaczynają działać (np. 3 workerów zaczyna działać w t0 = 1000
    workers = [t0] * k

    simulation_rows = []
    #symulacja
    for idx, row in queue_df.iterrows():
        #wybieramy indeks o najmniejszej wartości z workers (czyli workera, który będzie najszybciej gotowy do obsługi kolejnego zadania)
        worker_idx = int(np.argmin(workers))
        predicted_start_time = workers[worker_idx]
        #predykcja czasu wykonania się zadania w zależności od wybranego modelu
        predicted_duration = float(model.predict(row))

        #predykcja zakończenia wykonywania zadania
        predicted_end_time = predicted_start_time + predicted_duration
        #aktualizacja workera (np. t0 (1000s) zamieniamy na t0 + predicted_duration (1000s + 34s = 1034s)
        workers[worker_idx] = predicted_end_time

        #porównanie z rzeczywistym czasem wykonywania się zadań
        real_start_time = get_timestamp_seconds(row["startedAtDate"])
        real_end_time = get_timestamp_seconds(row["finishedAtDate"])
        real_duration = float(row["duration_seconds"])

        #historia wykonania się zadania
        simulation_rows.append({
            "queue_position": idx + 1,
            "worker_id": worker_idx + 1,

            "predicted_start_time": predicted_start_time,
            "predicted_end_time": predicted_end_time,
            "predicted_duration_seconds": predicted_duration,

            "real_start_time": real_start_time,
            "real_end_time": real_end_time,
            "real_duration_seconds": real_duration
        })

    #tworzenie dataframe z wyników
    result_df = pd.DataFrame(simulation_rows)
    #wykorzystanie funkcji enrich_details_dataframe czyli policzenie błędów i innych takich
    result_df = enrich_details_dataframe(result_df)

    #całkowity czas wykonania się kolejki
    predicted_queue_end = float(result_df["predicted_end_time"].max())
    #rzeczywisty czas wykonania się kolejki
    real_queue_end = float(result_df["real_end_time"].max())

    predicted_total_queue_time = predicted_queue_end - t0
    real_total_queue_time = real_queue_end - t0

    #liczenie błędów dla kolejki
    final_smape = smape_single(predicted_total_queue_time, real_total_queue_time)
    final_abs_error_seconds = abs(predicted_total_queue_time - real_total_queue_time)

    return {
        "mode": "offline",
        "k": k,
        "queue_start_time": t0,
        "predicted_end_time": predicted_queue_end,
        "real_end_time": real_queue_end,
        "predicted_total_queue_time_seconds": predicted_total_queue_time,
        "real_total_queue_time_seconds": real_total_queue_time,
        "absolute_error_seconds": final_abs_error_seconds,
        "smape": final_smape,
        "details_df": result_df
    }



# SYMULACJA ONLINE
def simulate_queue_online(model, df: pd.DataFrame, k: int):
    #symulacja online to wierna kopia symulacji offline z tą różnicą, że "douczamy" modele w trakcie działania
    queue_df = df.copy().reset_index(drop=True)

    t0 = get_timestamp_seconds(queue_df.iloc[0]["startedAtDate"])
    workers = [t0] * k

    simulation_rows = []

    for idx, row in queue_df.iterrows():
        # wybór najwcześniej dostępnego workera
        worker_idx = int(np.argmin(workers))
        predicted_start_time = workers[worker_idx]

        predicted_duration = float(model.predict(row))

        predicted_end_time = predicted_start_time + predicted_duration
        workers[worker_idx] = predicted_end_time

        real_start_time = get_timestamp_seconds(row["startedAtDate"])
        real_end_time = get_timestamp_seconds(row["finishedAtDate"])
        real_duration = float(row["duration_seconds"])

        #douczanie modeli
        model.update(row, real_duration)

        simulation_rows.append({
            "queue_position": idx + 1,
            "worker_id": worker_idx + 1,

            "predicted_start_time": predicted_start_time,
            "predicted_end_time": predicted_end_time,
            "predicted_duration_seconds": predicted_duration,

            "real_start_time": real_start_time,
            "real_end_time": real_end_time,
            "real_duration_seconds": real_duration
        })

    result_df = pd.DataFrame(simulation_rows)
    result_df = enrich_details_dataframe(result_df)

    predicted_queue_end = float(result_df["predicted_end_time"].max())
    real_queue_end = float(result_df["real_end_time"].max())

    predicted_total_queue_time = predicted_queue_end - t0
    real_total_queue_time = real_queue_end - t0

    final_smape = smape_single(predicted_total_queue_time, real_total_queue_time)
    final_abs_error_seconds = abs(predicted_total_queue_time - real_total_queue_time)

    return {
        "mode": "online",
        "k": k,
        "queue_start_time": t0,
        "predicted_end_time": predicted_queue_end,
        "real_end_time": real_queue_end,
        "predicted_total_queue_time_seconds": predicted_total_queue_time,
        "real_total_queue_time_seconds": real_total_queue_time,
        "absolute_error_seconds": final_abs_error_seconds,
        "smape": final_smape,
        "details_df": result_df
    }



def simulate(model, df: pd.DataFrame, k: int, mode="offline", feature_cols=None):
    if mode == "offline":
        return simulate_queue_offline(model, df, k)

    if mode == "online":
        return simulate_queue_online(model, df, k)

    raise ValueError("mode musi być 'offline' albo 'online'")



# ZAPIS SZCZEGÓŁÓW
def save_simulation_details(result: dict, output_path: str):
    details_df = result["details_df"].copy()
    details_df.to_csv(output_path, index=False)



# PRINT
def print_simulation_result(result: dict):
    print("\n=== WYNIK SYMULACJI ===")
    print(f"Tryb: {result['mode']}")
    print(f"Liczba workerów k: {result['k']}")
    print(f"Start kolejki: {seconds_to_datetime_str(result['queue_start_time'])}")
    print(f"Przewidywany koniec kolejki: {seconds_to_datetime_str(result['predicted_end_time'])}")
    print(f"Rzeczywisty koniec kolejki: {seconds_to_datetime_str(result['real_end_time'])}")
    print(f"Przewidywany czas całej kolejki [s]: {round(result['predicted_total_queue_time_seconds'], 4)}")
    print(f"Rzeczywisty czas całej kolejki [s]: {round(result['real_total_queue_time_seconds'], 4)}")
    print(f"Błąd bezwzględny [s]: {round(result['absolute_error_seconds'], 4)}")
    print(f"SMAPE [%]: {round(result['smape'], 4)}")