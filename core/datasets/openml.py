import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
)


def load_openml(
    name=None, data_id=None, target_col=None, version="active", random_state=None
):
    if data_id is not None:
        data = fetch_openml(data_id=data_id, as_frame=True, version=version)
    else:
        data = fetch_openml(name=name, as_frame=True, version=version)

    X = data.data
    y = data.target

    if target_col and target_col in X.columns:
        y = X[target_col]
        X = X.drop(columns=[target_col])

    if hasattr(y, "cat") or y.dtype == "object" or y.dtype.name == "category":
        y = y.astype("category").cat.codes

    for col in X.select_dtypes(include=["object", "category", "string"]).columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        non_null_orig = X[col].dropna()
        non_null_conv = converted.dropna()
        
        if len(non_null_orig) > 0 and len(non_null_conv) / len(non_null_orig) > 0.8:
            X[col] = converted

    config = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            config[col] = "continuous"
        else:
            config[col] = "categorical"

    def make_pipeline(model, resample):
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


if __name__ == "__main__":
    X, y, make_pipeline = load_openml_dataset(
        name="Telco-Customer-Churn", random_state=42
    )

    print(f"Original X shape: {X.shape}")
    print(f"Original y shape: {y.shape}")

    pipeline = make_pipeline()
    X_trans, y_trans = pipeline.fit_resample(X, y)

    print(f"Transformed X shape: {X_trans.shape}")
    print(f"Transformed y shape: {y_trans.shape}")