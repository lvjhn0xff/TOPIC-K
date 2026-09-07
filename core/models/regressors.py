import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def make_decision_tree_regressor(trial):
    params = {
        "criterion": trial.suggest_categorical(
            "dtr_criterion",
            ["squared_error", "absolute_error", "friedman_mse", "poisson"],
        ),
        "max_depth": trial.suggest_int(
            "dtr_max_depth", 2, 32, log=False
        ),
        "min_samples_split": trial.suggest_int(
            "dtr_min_samples_split", 2, 20
        ),
        "min_samples_leaf": trial.suggest_int(
            "dtr_min_samples_leaf", 1, 20
        ),
    }
    return DecisionTreeRegressor(**params)


def make_random_forest_regressor(trial):
    max_features = trial.suggest_categorical(
        "rfr_max_features", ["sqrt", "log2", "none"]
    )
    if max_features == "none":
        max_features = None

    params = {
        "n_estimators": trial.suggest_int(
            "rfr_n_estimators", 10, 500, step=10
        ),
        "criterion": trial.suggest_categorical(
            "rfr_criterion",
            ["squared_error", "absolute_error", "friedman_mse", "poisson"],
        ),
        "max_depth": trial.suggest_int(
            "rfr_max_depth", 2, 32, log=False
        ),
        "min_samples_split": trial.suggest_int(
            "rfr_min_samples_split", 2, 20
        ),
        "min_samples_leaf": trial.suggest_int(
            "rfr_min_samples_leaf", 1, 20
        ),
        "max_features": max_features,
    }
    return RandomForestRegressor(**params)


def make_linear_regressor(trial):
    params = {
        "fit_intercept": trial.suggest_categorical(
            "lr_fit_intercept", [True, False]
        ),
    }
    return LinearRegression(**params)


def make_svr(trial):
    params = {
        "C": trial.suggest_float(
            "svr_C", 0.001, 1000.0, log=True
        ),
        "kernel": trial.suggest_categorical(
            "svr_kernel", ["linear", "poly", "rbf", "sigmoid"]
        ),
        "gamma": trial.suggest_categorical(
            "svr_gamma", ["scale", "auto"]
        ),
        "degree": trial.suggest_int(
            "svr_degree", 2, 5
        ),
        "epsilon": trial.suggest_float(
            "svr_epsilon", 0.001, 10.0, log=True
        ),
    }
    return SVR(**params)


def make_k_neighbors_regressor(trial):
    params = {
        "n_neighbors": trial.suggest_int(
            "knr_n_neighbors", 1, 30
        ),
        "weights": trial.suggest_categorical(
            "knr_weights", ["uniform", "distance"]
        ),
        "metric": trial.suggest_categorical(
            "knr_metric", ["euclidean", "manhattan", "minkowski"]
        ),
    }
    return KNeighborsRegressor(**params)


def make_mlp_regressor(trial):
    n_layers = trial.suggest_int("mlpr_n_layers", 1, 3)
    hidden_layer_sizes = tuple(
        trial.suggest_int(f"mlpr_n_units_layer_{i}", 16, 128)
        for i in range(n_layers)
    )

    params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": trial.suggest_categorical(
            "mlpr_activation", ["tanh", "relu", "logistic"]
        ),
        "solver": trial.suggest_categorical(
            "mlpr_solver", ["sgd", "adam"]
        ),
        "alpha": trial.suggest_float(
            "mlpr_alpha", 1e-05, 0.1, log=True
        ),
        "learning_rate": trial.suggest_categorical(
            "mlpr_learning_rate", ["constant", "adaptive"]
        ),
        "max_iter": trial.suggest_int(
            "mlpr_max_iter", 100, 500
        ),
    }
    return MLPRegressor(**params)