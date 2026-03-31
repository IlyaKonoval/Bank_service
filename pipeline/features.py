import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self._is_fitted = False

    def fit(self, X, y=None):
        X_transformed = self._create_features(X.copy())
        self.feature_names_ = X_transformed.columns.tolist()
        self.scaler.fit(X_transformed)
        self._is_fitted = True
        return self

    def transform(self, X):
        X_transformed = self._create_features(X.copy())
        X_transformed = X_transformed.reindex(columns=self.feature_names_, fill_value=0)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_transformed),
            columns=self.feature_names_,
            index=X_transformed.index,
        )
        return X_scaled

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["LOAN_CLOSE_RATIO"] = np.where(
            df["LOAN_NUM_TOTAL"] > 0,
            df["LOAN_NUM_CLOSED"] / df["LOAN_NUM_TOTAL"],
            0.0,
        )

        df["OPEN_LOANS"] = df["LOAN_NUM_TOTAL"] - df["LOAN_NUM_CLOSED"]

        df["INCOME_PER_DEPENDANT"] = df["PERSONAL_INCOME"] / (df["DEPENDANTS"] + 1)

        df["INCOME_LOG"] = np.log1p(df["PERSONAL_INCOME"])

        df["AGE_INCOME_INTERACTION"] = df["AGE"] * df["INCOME_LOG"]

        df["LOAN_BURDEN"] = np.where(
            df["PERSONAL_INCOME"] > 0,
            df["OPEN_LOANS"] / df["PERSONAL_INCOME"] * 10000,
            0.0,
        )

        age_bins = [0, 25, 35, 45, 55, 100]
        age_labels = ["young", "adult", "middle", "senior", "elder"]
        df["AGE_GROUP"] = pd.cut(df["AGE"], bins=age_bins, labels=age_labels)
        age_dummies = pd.get_dummies(df["AGE_GROUP"], prefix="AGE", dtype=int)
        df = pd.concat([df, age_dummies], axis=1)
        df.drop("AGE_GROUP", axis=1, inplace=True)

        income_quantiles = [0, 12000, 20000, 35000, 60000, float("inf")]
        income_labels = ["low", "below_avg", "average", "above_avg", "high"]
        df["INCOME_GROUP"] = pd.cut(
            df["PERSONAL_INCOME"], bins=income_quantiles, labels=income_labels
        )
        income_dummies = pd.get_dummies(df["INCOME_GROUP"], prefix="INC", dtype=int)
        df = pd.concat([df, income_dummies], axis=1)
        df.drop("INCOME_GROUP", axis=1, inplace=True)

        return df
