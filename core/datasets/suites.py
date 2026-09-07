import openml
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


def load_cc18(dataset_identifier, target_col=None, random_state=None):
    suite = openml.study.get_suite(99)

    if isinstance(dataset_identifier, int) and dataset_identifier in suite.data:
        dataset_id = dataset_identifier
    else:
        dlist = openml.datasets.list_datasets(
            data_id=suite.data, output_format="dataframe"
        )
        if (
            isinstance(dataset_identifier, str)
            and dataset_identifier in dlist["name"].values
        ):
            dataset_id = int(
                dlist[dlist["name"] == dataset_identifier]["did"].values[0]
            )
        else:
            dataset_id = dataset_identifier

    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(
        target=(
            dataset.default_target_attribute
            if target_col is None
            else target_col
        ),
        dataset_format="dataframe",
    )

    if target_col and target_col in X.columns:
        y = X[target_col]
        X = X.drop(columns=[target_col])

    if hasattr(y, "cat") or y.dtype == "object" or y.dtype.name == "category":
        y = y.astype("category").cat.codes

    for col in X.select_dtypes(include=["object", "category", "string"]).columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        non_null_orig = X[col].dropna()
        non_null_conv = converted.dropna()
        if (
            len(non_null_orig) > 0
            and len(non_null_conv) / len(non_null_orig) > 0.8
        ):
            X[col] = converted

    config = {
        col: (
            "continuous"
            if pd.api.types.is_numeric_dtype(X[col])
            else "categorical"
        )
        for col in X.columns
    }

    def make_pipeline(model):
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

        return ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("resampler", SMOTE(random_state=random_state)),
            ("model", model)
        ])

    return X, y, make_pipeline


def load_ctr23(dataset_identifier, target_col=None, random_state=None):
    suite = openml.study.get_suite(353)

    if isinstance(dataset_identifier, int) and dataset_identifier in suite.data:
        dataset_id = dataset_identifier
    else:
        dlist = openml.datasets.list_datasets(
            data_id=suite.data, output_format="dataframe"
        )
        if (
            isinstance(dataset_identifier, str)
            and dataset_identifier in dlist["name"].values
        ):
            dataset_id = int(
                dlist[dlist["name"] == dataset_identifier]["did"].values[0]
            )
        else:
            dataset_id = dataset_identifier

    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(
        target=(
            dataset.default_target_attribute
            if target_col is None
            else target_col
        ),
        dataset_format="dataframe",
    )

    if target_col and target_col in X.columns:
        y = X[target_col]
        X = X.drop(columns=[target_col])

    if hasattr(y, "cat") or y.dtype == "object" or y.dtype.name == "category":
        y = y.astype("category").cat.codes

    for col in X.select_dtypes(include=["object", "category", "string"]).columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        non_null_orig = X[col].dropna()
        non_null_conv = converted.dropna()
        if (
            len(non_null_orig) > 0
            and len(non_null_conv) / len(non_null_orig) > 0.8
        ):
            X[col] = converted

    config = {
        col: (
            "continuous"
            if pd.api.types.is_numeric_dtype(X[col])
            else "categorical"
        )
        for col in X.columns
    }

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
    
        steps = [] 

        steps.append(("preprocessor", preprocessor))
        if resample: 
            steps.append(("resampler", SMOTE(random_state=42)))
        steps.append(("model", model))

        return ImbPipeline(steps=steps)

    return X, y, make_pipeline
