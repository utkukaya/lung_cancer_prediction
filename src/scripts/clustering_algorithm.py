import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.cluster import KMeans
from impyute.imputation.cs import fast_knn
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
import seaborn as sns
from collections import Counter
# from common_figures import Figures

class Clustering:
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
        y_prob = classifier.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)

        interp_tpr = np.interp(self.mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0

        return {'auc': auc_score, 'fpr': fpr, 'tpr': interp_tpr, 'thresholds': thresholds}

    def cross_val(self, classifier, X, y, param_grid=None, grid_search=True):
        scores, rocauc = [], []
        bests = [0, classifier]
        best_estimator = classifier
        kf = StratifiedKFold(10, random_state=0, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if grid_search is True:
                best_model_and_score = self.find_best(
                    classifier, X_train, y_train, 5, param_grid)

                best_estimator = best_model_and_score[2]
                if best_model_and_score[0] > bests[0]:
                    bests[0] = best_model_and_score[0]
                    bests[1] = best_estimator

            best_estimator.fit(X_train, y_train)
            y_pred = best_estimator.predict(X_test)
            scores.append(f1_score(y_test, y_pred, average="micro"))
            rocauc_result = self.find_roccurve(
                best_estimator, X_train, X_test, y_train, y_test)

            rocauc.append(rocauc_result)

        return {'f1_scores': scores, 'rocauc': rocauc, 'best_estimator': bests[1]}

    def train_test_model(self, X, y):
        pipe_lr = Pipeline([['sc', StandardScaler()], [
                           'clf', LogisticRegression(random_state=0)]])
        lr_grid = [{'clf__C': self.param_range,
                    'clf__penalty': ['l1', 'l2']}]
        lr_score = self.cross_val(pipe_lr, X, y, lr_grid, grid_search=False)
        return {'lr': lr_score}

    def return_metric_results(self, X, y):
        pipe_lr = Pipeline([['sc', StandardScaler()], [
                           'clf', LogisticRegression(random_state=0)]])
        pipe_rf = Pipeline([['sc', StandardScaler()], [
                           'clf', RandomForestClassifier(n_jobs=-1, random_state=0)]])
        pipe_svm = Pipeline(
            [['sc', StandardScaler()], ['clf', SVC(probability=True, random_state=0)]])

        lr_grid = [{'clf__C': self.param_range,
                    'clf__penalty': ['l1', 'l2']}]

        rf_grid = [{
            'clf__criterion': ['gini', 'entropy'],
            'clf__n_estimators': [10, 100, 500, 1000]
        }]

        svm_grid = [{'clf__C': self.param_range,
                    'clf__kernel': ['rbf', 'sigmoid']}]

        lr_score = self.cross_val(pipe_lr, X, y, lr_grid)
        rf_score = self.cross_val(pipe_rf, X, y, rf_grid)
        svm_score = self.cross_val(pipe_svm, X, y, svm_grid)
        return {'lr': lr_score, 'rf': rf_score, 'svm': svm_score}

    def k_means_clustering(self, X, y, labels, save_path):
        kmeans = KMeans(n_clusters=4, random_state=0)
        kmeans.fit(X)

        y_pred = kmeans.labels_
        cm = confusion_matrix(y, list(map(lambda x: x + 1, y_pred)))
        self.draw_confusion_matrix(cm, labels, save_path)
        TP = cm[1, 1]
        TN = cm[0, 0] + cm[2, 2]
        FP = cm[0, 1] + cm[2, 1]
        FN = cm[1, 0] + cm[1, 2]

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return {'accuracy': accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}

    def gmm_clustering(self, X, y, labels, save_path):
        total_random_state = 10
        total_accuracy = []
        total_precision = []
        total_recall = []
        total_f1_score = []
        all_cm = []
        for i in range(total_random_state):
            gmm = GaussianMixture(
                n_components=4, covariance_type='full', random_state=i)
            gmm.fit(X)  
            y_pred = gmm.predict(X)
            cm = confusion_matrix(y, list(map(lambda x: x + 1, y_pred)))
            all_cm.append(cm)
            TP = cm[1, 1]  
            TN = cm[0, 0] + cm[2, 2]  
            FP = cm[0, 1] + cm[2, 1] 
            FN = cm[1, 0] + cm[1, 2]  

            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1_score = 2 * (precision * recall) / (precision + recall)
            total_accuracy.append(accuracy)
            total_precision.append(precision)
            total_recall.append(recall)
            total_f1_score.append(f1_score)
        best_result_index = self.find_max_index(
            [total_accuracy, total_precision, total_recall, total_f1_score])
        self.draw_confusion_matrix(
            all_cm[best_result_index], labels, save_path)
        return {'accuracy': total_accuracy[best_result_index], "precision": total_precision[best_result_index], "recall": total_accuracy[best_result_index], "f1_score": total_f1_score[best_result_index]}

    def find_max_index(self, arrays):
        maxIndexes = []
        for array in arrays:
            maxIndexes.append(np.nanargmax(array))

        counter = Counter(maxIndexes)
        most_common = counter.most_common(1)
        most_common_value = most_common[0][0]
        return most_common_value

    def plotModelResult(self, true_y, pred_y):
        plt.scatter(true_y, pred_y, c=pred_y)
        plt.scatter(true_y, pred_y, c=true_y, marker='x',
                    cmap='Set1')  
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def draw_confusion_matrix(self, confusion_matrix, labels, save_path):
        sns.heatmap(confusion_matrix, annot=True, fmt='d',
                    xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.xlabel('Prediction Of Level Of Cancer')
        plt.ylabel('Level Of Cancer')
        plt.title('Confusion Matrix ' + "CDT_DI_FI_OHE")
        plt.savefig(save_path)
        # plt.close()
        plt.show()


    # def draw_plot(self, metric_score, save_path, paper_figure):
    #     figures = Figures()
    #     class_names = {'cdt_di_fi_ohe':'Approach K-Means For Classify Level Of Cancer'}

    #     classifier_names = ['accuracy', 'precision', 'recall', 'f1_score']

    #     scores = { 'cdt_di_fi_ohe':[metric_score[i.lower()] for i in classifier_names]}
    #     figures.draw_bar_plot(classifier_names, scores, class_names, grid_shape=(1, 1), save=save_path, paper_figure=paper_figure)

# clustering = Clustering()

# cdt_di_fi_ohe = pd.read_csv(
#     "/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/data/processed/dataframes/cdt_di_fi_ohe.csv")
# level_of_cancer = pd.read_csv(
#     "/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/data/processed/dataframes/level_of_cancer.csv").values.ravel()
# # targets = pd.read_csv("/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/data/processed/dataframes/targets.csv").values.ravel()
# # metrics_score_ohe_di_fi = model_and_evaluation.return_metric_results(ohe_di_fi.values, targets)
# labels = ["T1", "T2", "T3", "T4"]
# save_k_means = "/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/figures/paper_figures/png/clustering_k_means_CDT_DI_FI_OHE_confusion_matrix.png"
# save_k_means_scores = "/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/figures/paper_figures/png/clustering_k_means_CDT_DI_FI_OHE_scores.png"
# paper_figures_k_means_scores = "/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/figures/paper_figures/svg/clustering_k_means_CDT_DI_FI_OHE_scores.png"

# save_gmm_clustering = "/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/figures/paper_figures/png/clustering_gmm_CDT_DI_FI_OHE_confusion_matrix.png"
# save_gmm_clustering_scores = "/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/figures/paper_figures/png/clustering_gmm_CDT_DI_FI_OHE_scores.png"
# paper_figures_gmm_clustering_scores = "/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/figures/paper_figures/svg/clustering_gmm_CDT_DI_FI_OHE_confusion_scores.png"

# result_of_model_k_means = clustering.k_means_clustering(cdt_di_fi_ohe.values[0:70], level_of_cancer, labels, save_k_means)
# result_of_model_gmm = clustering.gmm_clustering(cdt_di_fi_ohe.values[0:70], level_of_cancer, labels, save_gmm_clustering)
# print(result_of_model_k_means)
# clustering.draw_plot(result_of_model_k_means, save_k_means_scores, paper_figures_k_means_scores)
# clustering.draw_plot(result_of_model_gmm, save_gmm_clustering_scores, paper_figures_gmm_clustering_scores)
# print(result_of_model_gmm)

# print(clustering.kMeansClustering(cdt_di_fi_ohe.values[0:70], level_of_cancer))
# clustering.plotModelResult(level_of_cancer, result_of_model)
# print(level_of_cancer)
