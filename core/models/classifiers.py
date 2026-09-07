import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def make_decision_tree_classifier(trial):
    params = {
        "criterion": trial.suggest_categorical(
            "dt_criterion", ["gini", "entropy", "log_loss"]
        ),
        "max_depth": trial.suggest_int(
            "dt_max_depth", 2, 32, log=False
        ),
        "min_samples_split": trial.suggest_int(
            "dt_min_samples_split", 2, 20
        ),
        "min_samples_leaf": trial.suggest_int(
            "dt_min_samples_leaf", 1, 20
        ),
    }
    return DecisionTreeClassifier(**params)


def make_random_forest_classifier(trial):
    max_features = trial.suggest_categorical(
        "rf_max_features", ["sqrt", "log2", "none"]
    )
    if max_features == "none":
        max_features = None

    params = {
        "n_estimators": trial.suggest_int(
            "rf_n_estimators", 10, 500, step=10
        ),
        "criterion": trial.suggest_categorical(
            "rf_criterion", ["gini", "entropy", "log_loss"]
        ),
        "max_depth": trial.suggest_int(
            "rf_max_depth", 2, 32, log=False
        ),
        "min_samples_split": trial.suggest_int(
            "rf_min_samples_split", 2, 20
        ),
        "min_samples_leaf": trial.suggest_int(
            "rf_min_samples_leaf", 1, 20
        ),
        "max_features": max_features,
    }
    return RandomForestClassifier(**params)


def make_gaussian_nb(trial):
    params = {
        "var_smoothing": trial.suggest_float(
            "gnb_var_smoothing", 1e-11, 1e-07, log=True
        )
    }
    return GaussianNB(**params)


def make_logistic_regression(trial):
    params = {
        "C": trial.suggest_float(
            "lr_C", 0.0001, 10000.0, log=True
        ),
        "solver": trial.suggest_categorical(
            "lr_solver", ["lbfgs", "liblinear", "saga"]
        ),
        "penalty": trial.suggest_categorical(
            "lr_penalty", ["l1", "l2", "elasticnet"]
        ),
        "max_iter": trial.suggest_int(
            "lr_max_iter", 100, 1000
        ),
    }
    return LogisticRegression(**params)


def make_svc(trial):
    params = {
        "C": trial.suggest_float(
            "svc_C", 0.001, 1000.0, log=True
        ),
        "kernel": trial.suggest_categorical(
            "svc_kernel", ["linear", "poly", "rbf", "sigmoid"]
        ),
        "gamma": trial.suggest_categorical(
            "svc_gamma", ["scale", "auto"]
        ),
        "degree": trial.suggest_int(
            "svc_degree", 2, 5
        ),
    }
    return SVC(**params)


def make_k_neighbors_classifier(trial):
    params = {
        "n_neighbors": trial.suggest_int(
            "knn_n_neighbors", 1, 30
        ),
        "weights": trial.suggest_categorical(
            "knn_weights", ["uniform", "distance"]
        ),
        "metric": trial.suggest_categorical(
            "knn_metric", ["euclidean", "manhattan", "minkowski"]
        ),
    }
    return KNeighborsClassifier(**params)


def make_mlp_classifier(trial):
    n_layers = trial.suggest_int("mlp_n_layers", 1, 3)
    hidden_layer_sizes = tuple(
        trial.suggest_int(f"mlp_n_units_layer_{i}", 16, 128)
        for i in range(n_layers)
    )

    params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": trial.suggest_categorical(
            "mlp_activation", ["tanh", "relu", "logistic"]
        ),
        "solver": trial.suggest_categorical(
            "mlp_solver", ["sgd", "adam"]
        ),
        "alpha": trial.suggest_float(
            "mlp_alpha", 1e-05, 0.1, log=True
        ),
        "learning_rate": trial.suggest_categorical(
            "mlp_learning_rate", ["constant", "adaptive"]
        ),
        "max_iter": trial.suggest_int(
            "mlp_max_iter", 100, 500
        ),
        "verbose" : True
    }
    return MLPClassifier(**params)