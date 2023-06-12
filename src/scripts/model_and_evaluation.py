import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd

class ModelAndEvaluation:
    def __init__(self) -> None:
        self.param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.mean_fpr = np.linspace(0, 1, 100)

    def find_best(self, classifier, X, y, cv, param_grid):
        gs = GridSearchCV(estimator=classifier,
                        param_grid=param_grid,
                        scoring='f1_micro',
                        cv=cv,
                        n_jobs=-1)
        gs = gs.fit(X, y)
        return (gs.best_score_, gs.best_params_, gs.best_estimator_) 

    def find_roccurve(self, classifier, X_train, X_test, y_train, y_test):
        """
        Finding roc_curve and Auc score for each classifier
        """
        classifier.fit(X_train, y_train)
        y_prob = classifier.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)

        interp_tpr = np.interp(self.mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0

        return {'auc':auc_score, 'fpr':fpr, 'tpr':interp_tpr, 'thresholds':thresholds}

    def cross_val(self, classifier, X, y, param_grid=None, grid_search=True):
        scores, rocauc = [], []
        bests = [0, classifier]
        best_estimator = classifier
        kf = StratifiedKFold(10, random_state=0, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            if grid_search is True:
                best_model_and_score = self.find_best(classifier, X_train, y_train, 5, param_grid)
                
                best_estimator = best_model_and_score[2]
                if best_model_and_score[0] > bests[0]:
                    bests[0] = best_model_and_score[0]
                    bests[1] = best_estimator
                
            best_estimator.fit(X_train, y_train)
            y_pred = best_estimator.predict(X_test)
            scores.append(f1_score(y_test, y_pred, average="micro"))
            rocauc_result = self.find_roccurve(best_estimator, X_train, X_test, y_train, y_test)

            rocauc.append(rocauc_result)
        
        return {'f1_scores':scores, 'rocauc':rocauc, 'best_estimator':bests[1]}
        
    def train_test_model(self, X, y):
        print(X)
        pipe_lr = Pipeline([['sc', StandardScaler()], ['clf', LogisticRegression(random_state=0)]])
        lr_grid = [{'clf__C': self.param_range,
                    'clf__penalty': ['l1','l2']}]
        lr_score = self.cross_val(pipe_lr, X, y, lr_grid, grid_search = False)
        return {'lr':lr_score}


    def return_metric_results(self, X, y):
        pipe_lr = Pipeline([['sc', StandardScaler()], ['clf', LogisticRegression(random_state=0)]])
        pipe_rf = Pipeline([['sc', StandardScaler()], ['clf', RandomForestClassifier(n_jobs=-1, random_state=0)]])
        pipe_svm = Pipeline([['sc', StandardScaler()], ['clf', SVC(probability=True, random_state=0)]])

        lr_grid = [{'clf__C': self.param_range,
                    'clf__penalty': ['l1','l2']}]
        
        rf_grid = [{
            'clf__criterion': ['gini', 'entropy'],
            'clf__n_estimators': [10, 100, 500, 1000]
        }]

        svm_grid = [{'clf__C': self.param_range,
                    'clf__kernel': ['rbf','sigmoid']}]

        lr_score = self.cross_val(pipe_lr, X, y, lr_grid)
        rf_score = self.cross_val(pipe_rf, X, y, rf_grid)
        svm_score = self.cross_val(pipe_svm, X, y, svm_grid)
        return {'lr':lr_score, 'rf':rf_score, 'svm':svm_score}

    
# model_and_evaluation = ModelAndEvaluation()

# ohe_di_fi = pd.read_csv("/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/data/processed/dataframes/ohe_di_fi.csv")

# targets = pd.read_csv("/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/data/processed/dataframes/targets.csv").values.ravel()

# metrics_score_ohe_di_fi = model_and_evaluation.return_metric_results(ohe_di_fi.values, targets)
# print(metrics_score_ohe_di_fi)