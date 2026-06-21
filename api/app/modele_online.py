import pandas as pd

from river import compose, preprocessing, tree


BASE_FEATURE_COLUMNS = [
    "pixel_count",
    "compression_ratio",
    "complexity",
    "model_group",
]

TARGET_COLUMN = "duration_seconds"


def prepare_online_training_dataframe(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Przygotowuje dane do trenowania modelu
    Hoeffding Tree.
    """

    data = df[
        BASE_FEATURE_COLUMNS + [TARGET_COLUMN]
    ].dropna().copy()

    data = data[
        data[TARGET_COLUMN] > 0
    ].copy()

    return data.reset_index(drop=True)


def prepare_river_features(
    row: pd.Series
) -> dict:
    """
    Zamienia rekord pandas na słownik
    wymagany przez bibliotekę River.
    """

    return {
        "pixel_count": float(
            row["pixel_count"]
        ),
        "compression_ratio": float(
            row["compression_ratio"]
        ),
        "complexity": float(
            row["complexity"]
        ),
        "model_group": str(
            row["model_group"]
        ),
    }


class HoeffdingTreeModel:
    """
    Model Hoeffding Tree przeznaczony do
    predykcji czasu wykonania zadań.

    Model jest początkowo trenowany na danych
    historycznych, a następnie może być
    aktualizowany przyrostowo za pomocą
    nowych zakończonych zadań.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        grace_period: int = 500,
        delta: float = 1e-2,
        max_depth: int = 20
    ):
        self.train_df = (
            prepare_online_training_dataframe(
                train_df
            )
        )

        self.model = (
            (
                compose.Select("model_group")
                | preprocessing.OneHotEncoder()
            )
            + compose.Select(
                "pixel_count",
                "compression_ratio",
                "complexity"
            )
            | tree.HoeffdingTreeRegressor(
                grace_period=grace_period,
                delta=delta,
                max_depth=max_depth
            )
        )

        for _, row in self.train_df.iterrows():
            features = prepare_river_features(
                row
            )

            duration = float(
                row[TARGET_COLUMN]
            )

            self.model.learn_one(
                features,
                duration
            )

    def predict(
        self,
        row: pd.Series
    ) -> float:
        features = prepare_river_features(
            row
        )

        prediction = self.model.predict_one(
            features
        )

        return float(prediction)

    def update(
        self,
        row: pd.Series,
        y_true: float
    ) -> None:
        features = prepare_river_features(
            row
        )

        self.model.learn_one(
            features,
            float(y_true)
        )