# importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from collections import OrderedDict
import sys
from impyute.imputation.cs import fast_knn

# To get data
data = pd.read_csv('lung_cancer.txt', sep="\t", encoding = "ISO-8859-1")
targets = np.ravel(data.iloc[:,0])
targets = np.where(targets == 'kontrol', 0, 1)
features = np.array(data.iloc[:,2:13])
columns = data.columns[2:13]

# Implementing one-hot encoding and Knn imputation for the option A
dummies_data_a = pd.get_dummies(data.iloc[:,2:13])
dummies_columns_a = dummies_data_a.columns

for clm in columns:
    dummies_data_a.loc[data[clm].isnull(), dummies_data_a.columns.str.startswith("{}_".format(clm))] = np.nan

sys.setrecursionlimit(100000) #Increase the recursion limit of the OS
imputed_knn_a = np.round(fast_knn(np.array(dummies_data_a), k=30)).astype('int')

# 3D PCA plots
sc = StandardScaler()
std_data = sc.fit_transform(imputed_knn_a)
pca = PCA(n_components=3)
pca_form = pca.fit_transform(std_data)
only_cancer = pca_form[targets == 1]
colors = ('orange','blue', 'green', 'brown', 'red', 'purple', 'gray')
markers = ['o','s','X','v','<','P','>']

def plot3d_lung(column_num, name, only_cancer=only_cancer, colors=colors, markers=markers):
    column = np.array(data.iloc[:,column_num]).astype('str')[targets == 1]
    unique_val = np.unique(column)

    plt.figure(figsize=(9,8))
    ax = plt.axes(projection="3d")
    for val, m, clr in zip(unique_val, markers, colors[:len(unique_val)]):
        for cor, v  in zip(only_cancer, column):
            if v == val:
                if v == 'nan':
                    continue
                else:
                    ax.scatter(cor[0], cor[1], cor[2], marker=m, c=clr, label=val, s=30)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title(name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Columns 
plot3d_lung(13, 'GTE3')
plot3d_lung(14, 'Gn')
plot3d_lung(15, 'GmE')
plot3d_lung(16, 'GESKtumoryeriiii3kolon')
plot3d_lung(17, 'GESKDiferansiyasyon')
plot3d_lung(18, 'ESKperinörinvdegerikolon')