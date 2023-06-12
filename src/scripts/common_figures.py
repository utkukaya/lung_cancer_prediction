import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.patches as mpatches
from sklearn.metrics import auc
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import shap


class Figures:
    def __init__(self) -> None:
        self.clr = ('orange', 'blue', 'green', 'purple', 'red', 'purple')
        self.mean_fpr = np.linspace(0, 1, 100)

    def plot_decision_boundaries(self, X, y, classifiers, classifier_titles, actual_labels=None,
                                 grid_shape=(1, 3), title=None, save=None, paper_figure=None):

        sc = StandardScaler()
        X_std = sc.fit_transform(X)

        pca2d = PCA(n_components=2, random_state=0)
        X_pca = pca2d.fit_transform(X_std)

        fig, ax = plt.subplots(nrows=grid_shape[0], ncols=grid_shape[1],
                               figsize=(grid_shape[1]*8, grid_shape[0]*8))

        if grid_shape == (1, 1):
            ax = [ax]

        for idx, cls in enumerate(classifiers):
            cls.fit(X_pca, y)
            plot_decision_regions(X_pca, y, cls, ax=ax[idx], markers='so')

            if actual_labels is not None:
                handles, labels = ax[idx].get_legend_handles_labels()
                ax[idx].legend(handles, actual_labels,
                               framealpha=0.3, scatterpoints=1)

            ax[idx].axis('off')
            ax[idx].set_title(classifier_titles[idx])

        if title is not None:
            fig.suptitle(title,
                         y=0.99)
        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches="tight")

        if paper_figure is not None:
            plt.savefig(paper_figure, bbox_inches="tight")

        plt.show()

    def create_table(self, columns, X, y, f_values, p_values, title=None, save=None, paper_figure=None):
        forest = RandomForestClassifier(n_estimators=10000,
                                        random_state=0,
                                        n_jobs=-1)

        forest.fit(X, y)
        importances = np.array(forest.feature_importances_)

        indices = np.argsort(f_values)[::-1]
        srtd_importances = np.round(
            importances[indices], 5).astype('str').reshape(-1, 1)
        srtd_fvalues = np.round(f_values[indices], 5).astype(
            'str').reshape(-1, 1)
        srtd_pvalues = np.round(p_values[indices], 10).astype(
            'str').reshape(-1, 1)
        srtd_columns = columns[indices]
        texts = np.hstack((srtd_importances, srtd_fvalues, srtd_pvalues))

        plt.figure(figsize=(5, 6))
        collabels = ('Feature Importances', 'F Values', 'P Values')
        table_ = plt.table(cellText=texts, rowLabels=srtd_columns,
                           colLabels=collabels, loc='center')
        plt.axis('off')
        plt.grid('off')
        table_.scale(1, 1.1)
        if title is not None:
            plt.suptitle(title, y=0.99)

        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches="tight")

        if paper_figure is not None:
            plt.savefig(paper_figure, bbox_inches="tight")

        plt.show()

    def draw_bar_plot(self, classifier_names, scores, class_names, grid_shape=(1, 3),
                      title=None, save=None, suptitle=True, paper_figure=None):
        fig, ax = plt.subplots(nrows=grid_shape[0], ncols=grid_shape[1],
                               figsize=(grid_shape[1]*8, grid_shape[0]*8))

        if grid_shape == (1, 1):
            ax = [ax]

        for idx, mthod in zip([i for i in range(grid_shape[1])], scores.keys()):
            means = [round(np.mean(i)*100, 2) for i in scores[mthod]]
            yerr = [[means[idx] - round(np.min(i)*100, 2), round(
                np.max(i)*100, 2) - means[idx]] for idx, i in enumerate(scores[mthod])]
            ax[idx].bar(classifier_names, means, yerr=np.array(yerr).T,
                        align='center', alpha=1.0, color=self.clr, capsize=5)
            ax[idx].set_yticks([i for i in range(0, 110, 10)])
            legends = []
            for i in range(len(classifier_names)):
                legends.append(mpatches.Patch(color=self.clr[i], label='{}: {}'.format(classifier_names[i],
                                                                                       means[i])))
            ax[idx].legend(handles=legends, loc='best')
            ax[idx].set_xlabel('Classifiers')
            ax[idx].set_ylabel('F1 Scores')
            if suptitle:
                ax[idx].title.set_text(class_names[mthod])

        if title is not None:
            fig.suptitle(title, y=0.99, fontsize=16)

        if save is not None:
            fig.savefig(save, dpi=300, bbox_inches="tight")

        if paper_figure is not None:
            plt.savefig(paper_figure, bbox_inches="tight")

        plt.show()

    def draw_rocauc_curve(self, classifier_names, scores, class_names, grid_shape=(1, 3),
                          title=None, save=None, suptitle=True, paper_figure=None):
        fig, ax = plt.subplots(nrows=grid_shape[0], ncols=grid_shape[1],
                               figsize=(grid_shape[1]*8, grid_shape[0]*8))

        if grid_shape == (1, 1):
            ax = [ax]

        for idx, key in zip([i for i in range(grid_shape[1])], scores.keys()):
            for result, name, clor in zip(scores[key], classifier_names, self.clr):
                tprs = [i['tpr'] for i in result]
                mean_tpr = np.array(tprs).mean(axis=0)
                mean_tpr[-1] = 1.0
                auc_mean = auc(self.mean_fpr, mean_tpr)
                ax[idx].plot(self.mean_fpr, mean_tpr, color=clor,
                             label='{}: {}'.format(name, round(auc_mean, 2)))

                tprs_upper = np.max(tprs, axis=0)
                tprs_lower = np.min(tprs, axis=0)
                ax[idx].fill_between(
                    self.mean_fpr,
                    tprs_lower,
                    tprs_upper,
                    color=clor,
                    alpha=0.2
                )

            ax[idx].plot([0, 1], [0, 1], color='black',
                         linestyle='--', alpha=0.5)
            ax[idx].legend(loc='best')
            ax[idx].set_xlabel('False Positive Rate')
            ax[idx].set_ylabel('True Positive Rate')
            if suptitle:
                ax[idx].title.set_text("{}".format(class_names[key]))

        if title is not None:
            fig.suptitle(title, y=0.99, fontsize=16)

        if save is not None:
            fig.savefig(save, dpi=300, bbox_inches="tight")

        if paper_figure is not None:
            plt.savefig(paper_figure, bbox_inches="tight")

        plt.show()

    def draw_dendogram(self, data, targets, column_number, title, only_cancer, save=None,
                       paper_figure=None):
        eliminated_nan_labels, eliminated_nan_pca = [], []
        label = np.array(data.iloc[:, column_number]
                         ).astype('str')[targets == 1]
        for lbl, pca_values in zip(label, only_cancer):
            if lbl == 'nan':
                continue
            eliminated_nan_labels.append(lbl)
            eliminated_nan_pca.append(pca_values)

        variables = ['X', 'Y', 'Z']
        df = pd.DataFrame(eliminated_nan_pca, columns=variables,
                          index=eliminated_nan_labels)

        row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
                                columns=eliminated_nan_labels,
                                index=eliminated_nan_labels)

        row_clusters = linkage(
            df.values, method='complete', metric='euclidean')

        # make dendrogram black (part 1/2)
        # from scipy.cluster.hierarchy import set_link_color_palette
        # set_link_color_palette(['black'])

        row_dendr = dendrogram(row_clusters,
                               labels=eliminated_nan_labels,
                               # make dendrogram black (part 2/2)
                               # color_threshold=np.inf
                               )
        plt.tight_layout()
        plt.ylabel('Euclidean distance')
        plt.title('Hierarchical Clustering Cancer Labeled With {}'.format(title))
        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches='tight')

        if paper_figure is not None:
            plt.savefig(paper_figure, bbox_inches="tight")

        plt.show()

    def feature_importance_plot(self, X, y, columns, figsize=(15, 5), title=None,
                                save=None, paper_figure=None):
        forest = RandomForestClassifier(n_estimators=10000,
                                        random_state=0,
                                        n_jobs=-1)

        sc = StandardScaler()
        X_train_std = sc.fit_transform(X)

        forest.fit(X_train_std, y)
        importances = forest.feature_importances_

        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=figsize)

        if title is not None:
            plt.title(title)

        plt.bar(range(X_train_std.shape[1]),
                importances[indices],
                color='lightblue',
                align='center')

        plt.xticks(range(X_train_std.shape[1]),
                   columns[indices], rotation=90)
        plt.xlim([-1, X_train_std.shape[1]])
        plt.ylim([0, np.max(importances)*1.1])

        sns.despine()

        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches='tight')

        if paper_figure is not None:
            plt.savefig(paper_figure, bbox_inches="tight")

        plt.show()

    def feature_importance_plot_return_features(self, X, y, columns, figsize=(15, 5), title=None,
                                                save=None, paper_figure=None):
        forest = RandomForestClassifier(n_estimators=10000,
                                        random_state=0,
                                        n_jobs=-1)

        sc = StandardScaler()
        X_train_std = sc.fit_transform(X)

        forest.fit(X_train_std, y)
        importances = forest.feature_importances_

        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=figsize)

        if title is not None:
            plt.title(title)

        plt.bar(range(X_train_std.shape[1]),
                importances[indices],
                color='lightblue',
                align='center')

        plt.xticks(range(X_train_std.shape[1]),
                   columns[indices], rotation=90)
        plt.xlim([-1, X_train_std.shape[1]])
        plt.ylim([0, np.max(importances)*1.1])

        sns.despine()

        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches='tight')

        if paper_figure is not None:
            plt.savefig(paper_figure, bbox_inches="tight")

        plt.show()
        return columns[indices]

    def feature_importance_plot_shap(self, X, y, columns, figsize=(15, 5), title=None,
                                     save=None, paper_figure=None):

        forest = RandomForestClassifier(n_estimators=10000,
                                        random_state=0,
                                        n_jobs=-1)

        sc = StandardScaler()
        X_train_std = sc.fit_transform(X)

        forest.fit(X_train_std, y)
        explainer = shap.Explainer(forest, X)

        # SHAP değerlerini hesapla
        shap_values = explainer.shap_values(X)

        # Özellik önemini görselleştir
        # shap.summary_plot(shap_values, X, feature_names=columns)

        fig = shap.summary_plot(shap_values, X, show=False, feature_names=columns)
        plt.savefig(save)


# figures = Figures()
# main_path = "../.."
# df_path = f"/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/data/processed/dataframes"
# cdt_di_fi_ohe = pd.read_csv(f"{df_path}/cdt_di_fi_ohe.csv")
# targets = pd.read_csv(f"{df_path}/targets.csv").values.ravel()
# save = f"/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/figures/paper_figures/png/feature_importance_cdt_di_fi_ohe_shap.png"
# paper_figure = f"{main_path}/figures/paper_figures/svg/feature_importance_cdt_di_fi_ohe_shap.svg"
# importance_columns = figures.feature_importance_plot_shap(
#     cdt_di_fi_ohe.values, targets, cdt_di_fi_ohe.columns, save=save, paper_figure=paper_figure)

# save = f"/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/figures/paper_figures/png/feature_importance_ohe_di.png"
# paper_figure = f"/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/figures/paper_figures/svg/feature_importance_ohe_di.svg"
# figures.feature_importance_plot(cdt_di_fi_ohe.values, targets, cdt_di_fi_ohe.columns,
#                                 save=save, paper_figure=paper_figure)