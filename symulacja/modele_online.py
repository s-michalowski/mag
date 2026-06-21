import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from river import tree, compose, preprocessing


BASE_FEATURE_COLUMNS = [
    "pixel_count",
    "compression_ratio",
    "complexity",
    "model_group"
]

TARGET_COLUMN = "duration_seconds"



def prepare_online_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    #Przygotowuje dataframe do trenowania modeli

    data = df.copy()
    data = data[BASE_FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().copy()
    data = data[data[TARGET_COLUMN] > 0].copy()

    return data.reset_index(drop=True)



# 1. ONLINE LINEAR REGRESSION

class OnlineLinearRegressionModel:
    """
    Model działa w następujący sposób:
    - na początku trenuje klasyczną regresję liniową na danych historycznych,
    - używa tych samych cech co regresja offline:
        pixel_count,
        compression_ratio,
        complexity,
        model_group zakodowany przez one-hot encoding,
    - po każdym nowym zakończonym zadaniu dopisuje rekord do bazy,
    - następnie trenuje LinearRegression od nowa na całej powiększonej bazie.
    """

    def __init__(self, train_df: pd.DataFrame):
        self.train_df = prepare_online_training_dataframe(train_df)

        self.model = LinearRegression()
        self.feature_cols = None

        self._fit_model()

    def _prepare_features(self, df: pd.DataFrame):
        data = df.copy()

        data = data[BASE_FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().copy()
        data = data[data[TARGET_COLUMN] > 0].copy()

        data = pd.get_dummies(data, columns=["model_group"])

        feature_cols = [
            col for col in data.columns
            if col != TARGET_COLUMN
        ]

        X = data[feature_cols].copy()
        y = data[TARGET_COLUMN].copy()

        return X, y, feature_cols

    def _fit_model(self):
        X_train, y_train, feature_cols = self._prepare_features(self.train_df)

        self.feature_cols = feature_cols
        self.model.fit(X_train, y_train)

    def _prepare_row_for_prediction(self, row: pd.Series):
        row_df = row.to_frame().T

        row_df = row_df[BASE_FEATURE_COLUMNS].copy()
        row_df = pd.get_dummies(row_df, columns=["model_group"])

        for col in self.feature_cols:
            if col not in row_df.columns:
                row_df[col] = 0

        row_df = row_df[self.feature_cols].copy()

        return row_df

    def predict(self, row: pd.Series) -> float:
        X = self._prepare_row_for_prediction(row)

        pred = float(self.model.predict(X)[0])

        return pred

    def update(self, row: pd.Series, y_true: float):
        new_row = {
            "pixel_count": row["pixel_count"],
            "compression_ratio": row["compression_ratio"],
            "complexity": row["complexity"],
            "model_group": row["model_group"],
            "duration_seconds": float(y_true)
        }

        self.train_df = pd.concat(
            [self.train_df, pd.DataFrame([new_row])],
            ignore_index=True
        )

        self._fit_model()


# 2. HOEFFDING TREE
def prepare_river_features(row: pd.Series) -> dict:
    #Zamiana rekordu na słownik dla drzewa

    return {
        "pixel_count": float(row["pixel_count"]),
        "compression_ratio": float(row["compression_ratio"]),
        "complexity": float(row["complexity"]),
        "model_group": str(row["model_group"])
    }


class HoeffdingTreeModel:
    """
    Hoeffding Tree:
    - uczymy na pełnym zbiorze danych
    - podczas symulacji:
        predict(row)
        update(row, y_real)
    """
    def __init__(
            self,
            train_df: pd.DataFrame,
            #co ile drzewo decyduje się na podział
            grace_period: int = 500,
            #jak ostrożnie drzewo decyduje się na podział
            delta: float = 1e-2,
            #maksymalna głębokość drzewa
            max_depth: int = 20
    ):
        self.train_df = prepare_online_training_dataframe(train_df)


        self.model = (
                #bierzemy tylko model group i robimy one-hot-encoding tej kolumny ->
                #dołączamy resztę cech (compression_ratio + pixel_count) i wrzucamy je do treeRegressor
                (compose.Select("model_group") | preprocessing.OneHotEncoder())
                + compose.Select("pixel_count", "compression_ratio", "complexity") #
                | tree.HoeffdingTreeRegressor(
            grace_period=grace_period,
            delta=delta,
            max_depth=max_depth
        )
        )
        #trenowanie drzewa na danych treningowych
        for _, row in self.train_df.iterrows():
            X = prepare_river_features(row)
            y = float(row[TARGET_COLUMN])
            self.model.learn_one(X, y)

    def predict(self, row: pd.Series) -> float:
        X = prepare_river_features(row)
        pred = self.model.predict_one(X)
        return pred

    def update(self, row: pd.Series, y_true: float):
        X = prepare_river_features(row)
        self.model.learn_one(X, float(y_true))



# 3. CBR
class CBRModel:
    """
    Case-Based Reasoning:
    - baza przypadków startuje z train
    - normalize liczone na bazie train
    - predict:
         najpierw szuka sąsiadów w tym samym model_group
         jeśli ich za mało, to szukamy najlepiej pasujących rekordów dla całej bazy danych
    - update:
         jeśli retain=True, dodaje nowy case do bazy
         po dodaniu przelicza normalizację
    """

    def __init__(self, train_df: pd.DataFrame, k: int = 3, retain: bool = True):

        #k - liczba sąsiadów
        self.k = k
        #czy dopuszczamy do zapisu (czyli czy dopuszczamy "learning bazy wiedzy")
        self.retain_enabled = retain
        #czyszczenie bazy wiedzy
        self.train_df = prepare_online_training_dataframe(train_df)

        self._recompute_normalization()

    def _recompute_normalization(self):

        #normalizacja cech do przedziału 0-1 w celu lepszych predykcji modelu
        self.pixel_min = float(self.train_df["pixel_count"].min())
        self.pixel_max = float(self.train_df["pixel_count"].max())

        self.comp_min = float(self.train_df["compression_ratio"].min())
        self.comp_max = float(self.train_df["compression_ratio"].max())

        self.complexity_min = float(self.train_df["complexity"].min())
        self.complexity_max = float(self.train_df["complexity"].max())

        self.pixel_range = self.pixel_max - self.pixel_min
        self.comp_range = self.comp_max - self.comp_min
        self.complexity_range = self.complexity_max - self.complexity_min

        if self.pixel_range == 0:
            self.pixel_range = 1.0
        if self.comp_range == 0:
            self.comp_range = 1.0
        if self.complexity_range == 0:
            self.complexity_range = 1.0

        self.train_df = self.train_df.copy()
        self.train_df["pixel_norm"] = (
            (self.train_df["pixel_count"] - self.pixel_min) / self.pixel_range
        )
        self.train_df["comp_norm"] = (
            (self.train_df["compression_ratio"] - self.comp_min) / self.comp_range
        )
        self.train_df["complexity_norm"] = (
            (self.train_df["complexity"] - self.complexity_min) / self.complexity_range
        )

    def _normalize_row(self, row: pd.Series) -> pd.Series:
        #również normalizacja tylko dla jednego rekordu
        row_copy = row.copy()

        row_copy["pixel_norm"] = (
            (float(row_copy["pixel_count"]) - self.pixel_min) / self.pixel_range
        )
        row_copy["comp_norm"] = (
            (float(row_copy["compression_ratio"]) - self.comp_min) / self.comp_range
        )
        row_copy["complexity_norm"] = (
            (float(row_copy["complexity"]) - self.complexity_min) / self.complexity_range
        )
        return row_copy

    @staticmethod
    def _distance(row_x: pd.Series, row_y: pd.Series) -> float:
        #odległość euklidesowa
        return float(np.sqrt(
            (float(row_x["pixel_norm"]) - float(row_y["pixel_norm"])) ** 2 +
            (float(row_x["comp_norm"]) - float(row_y["comp_norm"])) ** 2 +
            (float(row_x["complexity_norm"]) - float(row_y["complexity_norm"])) ** 2
        ))

    def _get_neighbors(self, candidates_df: pd.DataFrame, row_norm: pd.Series):
        distances = []
        #dla każdego rekordu w bazie wiedzy liczymy odległość (euklidesową)
        for _, candidate in candidates_df.iterrows():
            dist = self._distance(row_norm, candidate)
            distances.append((dist, candidate))
        #zwracamy najbliższych sąsiadów (najbardziej podobne rekordy)
        distances.sort(key=lambda x: x[0])
        #wybieramy k sąsiadów z tych najbliższych
        effective_k = min(self.k, len(distances))
        neighbors = [item[1] for item in distances[:effective_k]]
        #zwracamy sąsiadów
        return neighbors

    def predict(self, row: pd.Series) -> float:
        row_norm = self._normalize_row(row)

        same_model_cases = self.train_df[
            self.train_df["model_group"] == row["model_group"]
        ]

        if len(same_model_cases) >= self.k:
            candidates = same_model_cases
        else:
            candidates = self.train_df

        neighbors = self._get_neighbors(candidates, row_norm)
        #średni czas wykonania się zadania ze znalezionych sąsiadów
        pred = float(np.mean([float(n[TARGET_COLUMN]) for n in neighbors]))

        return pred

    def update(self, row: pd.Series, y_true: float):
        if not self.retain_enabled:
            return

        new_row = {
            "pixel_count": row["pixel_count"],
            "compression_ratio": row["compression_ratio"],
            "complexity": row["complexity"],
            "model_group": row["model_group"],
            "duration_seconds": float(y_true)
        }

        self.train_df = pd.concat(
            [self.train_df, pd.DataFrame([new_row])],
            ignore_index=True
        )

        self._recompute_normalization()