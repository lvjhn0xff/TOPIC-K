import json
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
)


def load_custom(csv_path, json_path, target_col):
    df = pd.read_csv(csv_path)
    with open(json_path, "r") as f:
        config = json.load(f)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    def make_pipeline(model, resample=True):
        cat_cols = [
            k for k, v in config.items() 
            if v == "categorical" and k in X.columns
        ]
        ord_cols = [
            k for k, v in config.items() 
            if v == "ordinal" and k in X.columns
        ]
        num_cols = [
            k for k, v in config.items() 
            if v == "continuous" and k in X.columns
        ]

        cat_transformer = ImbPipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        ord_transformer = ImbPipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder()),
        ])

        num_transformer = ImbPipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", cat_transformer, cat_cols),
                ("ord", ord_transformer, ord_cols),
                ("num", num_transformer, num_cols),
            ]
        )

        steps = [] 

        steps.append(("preprocessor", preprocessor))
        if resample: 
            steps.append(("resampler", SMOTE(random_state=42)))
        steps.append(("model", model))

        return ImbPipeline(steps=steps)

    return X, y, make_pipeline
