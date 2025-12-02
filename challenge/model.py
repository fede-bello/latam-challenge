import pandas as pd

from typing import Tuple, Union, List
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

THRESHOLD_IN_MINUTES = 15

MODEL_PATH = "challenge/model.pkl"


class DelayModel:
    _TOP_10_FEATURES = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]
    _THRESHOLD_IN_MINUTES = 15

    def __init__(self):
        self._model = None  # Model should be saved in this attribute.

    def _get_min_diff(self, data):
        fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def _get_top_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.reindex(columns=self._TOP_10_FEATURES, fill_value=0)

    def _save_model(self, path: str) -> None:
        """
        Save the trained model to a file.

        Args:
            path (str): The file path where the model will be saved.
        """

        joblib.dump(self._model, path)

    def _load_model(self, path: str) -> None:
        """
        Load a trained model from a file.

        Args:
            path (str): The file path from where the model will be loaded.
        """

        self._model = joblib.load(path)

    def _get_target_column(
        self, data: pd.DataFrame, target_column: str
    ) -> pd.DataFrame:
        data["min_diff"] = data.apply(self._get_min_diff, axis="columns")
        data[target_column] = np.where(
            data["min_diff"] > self._THRESHOLD_IN_MINUTES, 1, 0
        )
        return pd.DataFrame(data[target_column])

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis="columns",
        )
        features = self._get_top_features(features)
        if not target_column:
            return features

        target = self._get_target_column(data, target_column)
        return features, target

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        target = target.squeeze()
        X_train, _, y_train, _ = train_test_split(
            features, target, test_size=0.33, random_state=42
        )
        n_y0, n_y1 = len(y_train[y_train == 0]), len(y_train[y_train == 1])

        self._model = LogisticRegression(
            class_weight={1: n_y0 / len(y_train), 0: n_y1 / len(y_train)}
        )
        self._model.fit(X_train, y_train)

        self._save_model(MODEL_PATH)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            self._load_model(MODEL_PATH)
        predictions = self._model.predict(features).tolist()
        return predictions
