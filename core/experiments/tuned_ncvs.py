from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, 
    RepeatedKFold, KFold 
)
from imblearn.pipeline import Pipeline
from collections import Counter
import numpy as np 
import optuna

from sklearn.metrics import (
    # Classification
    accuracy_score, 
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef, 
    precision_score,
    recall_score, 
    f1_score, 
    average_precision_score, 
    roc_auc_score, 
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve, 
    roc_curve, 

    # Regression 
    mean_absolute_percentage_error, 
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percentage_error, 
    r2_score
)


class TunedNCVS: 
    def __init__(
        self, task_type,
        X, 
        y, 
        make_model,
        random_state=42
    ):
        self.task_type = task_type 
        self.X = X 
        self.y = y 
        self.outer_repeats = 5
        self.outer_splits = 5 
        self.inner_splits = 5
        self.make_model = make_model
        self.base_rskf = None 
        self.random_state = random_state

    def define_model(self): 
        return LogisticRegression(max_iter=1000, verbose=10)

    def describe_dataset(self): 
        print(":: Describing dataset.") 
        print("\tDataset Shape :", self.X.shape)
        print("\tLabel Shape   :", self.y.shape)

    def report_label_distribution(self, items, indent=""): 
        if self.task_type == "classification": 
            frequency = dict(Counter(items))
            for key in sorted(frequency.keys()):
                print(f"{indent}\t{key} = {frequency[key]}")
            return frequency
        elif self.task_type == "regression": 
            arr = np.array(items)
            percentiles = np.percentile(arr, [0, 25, 50, 75, 100])
            print(
                f"{indent}\tPercentiles (Min, 25%, Median, 75%, Max): "
                f"{[round(float(x), 2) for x in list(percentiles)]}"
            )
            hist, bin_edges = np.histogram(arr, bins=10)
            print(f"{indent}\tHistogram Counts: {[int(x) for x in list(hist)]}")
            print(
                f"{indent}\tBin Edges: "
                f"{[round(float(x), 2) for x in list(bin_edges)]}"
            )
            return percentiles, hist, bin_edges
        else: 
            raise Exception(f"Unknown task type: {self.task_type}")

    def configure_structure(self):
        BASE_FOLDER = None 
        SUB_FOLDER = None


        if self.task_type == "classification": 
            BASE_FOLDER = RepeatedStratifiedKFold 
            SUB_FOLDER = StratifiedKFold
        elif self.task_type == "regression":
            BASE_FOLDER = RepeatedKFold
            SUB_FOLDER = KFold
        else: 
            raise Exception(f"Unknown task type: {self.task_type}")

        self.base_rskf = BASE_FOLDER(
            n_repeats=self.outer_repeats,
            n_splits=self.outer_splits,
            random_state=self.random_state
        )   

        batch_no = 1 
        for main_index, holdout_set in self.base_rskf.split(self.X, self.y): 
            print(f":: Running Batch #{batch_no}") 
            
            # Splitting to main and hold-out set. 
            print(f"\t:: Splitting to main and hold-out set.")
            X_main, y_main = self.X.iloc[main_index], self.y.iloc[main_index] 
            X_holdout, y_holdout = self.X.iloc[holdout_set], self.y.iloc[holdout_set]

            # Report distribution. 
            print("\t\t\t:: Reporting label distribution [main]")
            main_ld = self.report_label_distribution(y_main, indent="\t\t")  

            print("\t\t\t:: Reporting label distribution [holdout]")
            holdout_ld = self.report_label_distribution(y_holdout, indent="\t\t")  
            
            # Splitting to sub-folds. 
            sub_rskf = SUB_FOLDER(
                n_splits=self.inner_splits,
                random_state=self.random_state,
                shuffle=True
            )

            # Run sub-fold. 
            def objective(trial): 
                score = self.run_subfold(trial, sub_rskf, X_main, y_main)
                return score 

            # Run study.
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler) 
            study.optimize(objective, n_trials=300)

            batch_no += 1

    def run_subfold(self, trial, sub_rskf, X_main, y_main):
        # Collect scores from the current batch. 
        train_results = []
        test_results = []

        # Split into train_index and val_index.
        sub_fold = 1
        print(f"\t:: Splitting into sub-folds.")
        for train_index, val_index in sub_rskf.split(X_main, y_main): 
            print(f"\t\t:: Sub-Fold {sub_fold}")

            # Split to Train and Validation Set
            print(f"\t\t\t:: Splitting to main and hold-out set.")
            X_train, y_train = X_main.iloc[train_index], y_main.iloc[train_index] 
            X_val, y_val = X_main.iloc[val_index], y_main.iloc[val_index] 

            print("\t\t\t:: Reporting label distribution [train]")
            train_ld = self.report_label_distribution(y_train, indent="\t\t\t")  

            print("\t\t\t:: Reporting label distribution [val]")
            val_ld = self.report_label_distribution(y_val, indent="\t\t\t")   

            # Defining model
            print(f"\t\t\t:: Instantiating model.")
            model = self.define_model(trial)                

            # Train model. 
            print(f"\t\t\t:: Training model.")
            pipeline = self.train_model(model, X_train, y_train)

            # Evaluate model. 
            print(f"\t\t\t:: Evaluating model on train set.")
            train_score = self.evaluate_model_main(pipeline, X_train, y_train)

            print(f"\t\t\t:: Evaluating model on validation set.")
            test_score = self.evaluate_model_main(pipeline, X_val, y_val)  

            # Register scores.
            print(f"\t\t\t:: Registering scores.")
            train_results.append(train_score)
            test_results.append(test_score)

            # Move to next fold.
            sub_fold += 1

        # Return average of test scores. 
        average_test_score = sum(test_results) / len(test_results)
        
        return average_test_score


    def make_pipeline(self, model): 
        resample = self.task_type == "classification"
        pipeline = self.make_model(model, resample=resample)
        return pipeline

    def train_model(self, model, X, y): 
        pipeline = self.make_pipeline(model)
        pipeline.fit(X, y)
        return pipeline 

    def evaluate_model_main(self, pipeline, X, y):
        if self.task_type == "classification": 
            y_pred = pipeline.predict(X) 
            score = balanced_accuracy_score(y, y_pred) 
            print(f"\t\t\t\tBalanced Accuracy Score : {score}")
            return score 
        elif self.task_type == "regression": 
            y_pred = pipeline.predict(X) 
            score = mean_squared_error(y, y_pred) 
            print(f"\t\t\t\tMean Squared Error      : {score}")
            return score
        else:
            raise Exception(f"Unknown task type: {self.task_type}")

    def run(self): 
        self.describe_dataset()
        self.configure_structure()


   