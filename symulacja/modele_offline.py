import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


BASE_FEATURE_COLUMNS = [
    "pixel_count",
    "compression_ratio",
    "complexity",
    "model_group"
]

TARGET_COLUMN = "duration_seconds"


def prepare_features(df: pd.DataFrame):
    #Przygotowuje cechy do trenowania modeli offline

    data = df.copy()

    data = data[BASE_FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().copy()
    data = data[data[TARGET_COLUMN] > 0].copy()

    #one-hot encoding - nazwy modeli zamieniamy na liczby, które modele będą rozróżniać
    data = pd.get_dummies(data, columns=["model_group"])
    #wszystkie kolumny zapisujemy do feature_cols (razem z tymi nowymi zone-hot encodowanymi)
    feature_cols = [col for col in data.columns if col != TARGET_COLUMN]

    X = data[feature_cols].copy()
    y = data[TARGET_COLUMN].copy()

    return X, y, feature_cols


def prepare_features_for_prediction(df: pd.DataFrame, feature_cols):
    #Przygotowuje cechy do predykcji.
    data = df.copy()

    data = data[BASE_FEATURE_COLUMNS].copy()
    data = pd.get_dummies(data, columns=["model_group"])

    #upewnienie się, że wszystkie kolumny są dodane
    for col in feature_cols:
        if col not in data.columns:
            data[col] = 0

    # usuwa nadmiarowe kolumny, jeśli w predykcji pojawiły się nowe model_group
    data = data[feature_cols].copy()

    return data


# TRENING MODELI

def train_baseline_group_median(train_df: pd.DataFrame):
    #Trening baseline'u
    data = train_df[["model_group", TARGET_COLUMN]].dropna().copy()
    data = data[data[TARGET_COLUMN] > 0].copy()

    baseline_model = data.groupby("model_group")[TARGET_COLUMN].median()

    feature_cols = ["model_group"]

    return baseline_model, feature_cols

def train_linear_regression(train_df: pd.DataFrame):
    #Trening regresji liniowej

    X, y, feature_cols = prepare_features(train_df)

    model = LinearRegression()
    model.fit(X, y)

    return model, feature_cols


def train_random_forest(train_df: pd.DataFrame):
    #Trening Random Forest

    X, y, feature_cols = prepare_features(train_df)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=1,
        n_jobs=-1
    )
    model.fit(X, y)

    return model, feature_cols


def train_neural_network(train_df: pd.DataFrame):
    #Trening sieci neuronowej

    X, y, feature_cols = prepare_features(train_df)

    #Skalowanie danych bo pixele w milionach a compression ratio w jednostkach (np. pixel_count = 3 920 400 (1980x1980) a compression ratio 1.6)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=400,
        random_state=1
    )
    model.fit(X_scaled, y)

    return model, feature_cols, scaler


def predict_single(model, row: pd.Series, feature_cols, scaler=None):
    #Predykcja dla pojedynczego rekordu.
    #Dla NN przekazujemy scaler, dla regresji i RF scaler=None.
    #Dla baseline sprawdzamy medianę dla poznanej model_group

    if isinstance(model, pd.Series):
        model_group = row["model_group"]
        return float(model.loc[model_group])

    row_df = row.to_frame().T
    X = prepare_features_for_prediction(row_df, feature_cols)

    if scaler is not None:
        X = scaler.transform(X)

    pred = float(model.predict(X)[0])

    return pred


