import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits import mplot3d
from collections import OrderedDict

data = pd.read_csv('lung_cancer.txt', sep="\t", encoding = "ISO-8859-1")
targets = np.ravel(data.iloc[:,0])
targets = np.where(targets == 'kontrol', 0, 1)

sparsed_features = pd.get_dummies(data.iloc[:,2:13], dummy_na=True, dtype='float').values

sc = StandardScaler()
std_data = sc.fit_transform(sparsed_features)
pca = PCA(n_components=3)
pca_form = pca.fit_transform(std_data)
only_cancer = pca_form[targets == 1]
colors = ('orange','blue', 'green', 'brown', 'red', 'purple', 'gray')
markers = ['o','s','X','v','<','P','>']

def pca_3dplot_lung(column_num, name, only_cancer=only_cancer, colors=colors, markers=markers):
    column = np.array(data.iloc[:,column_num]).astype('str')[targets == 1]
    unique_val = np.unique(column)

    plt.figure(figsize=(9,8))
    ax = plt.axes(projection="3d")
    for val, m, clr in zip(unique_val, markers, colors[:len(unique_val)]):
        for cor, colm  in zip(only_cancer, column):
            if colm == val:
                if colm == 'nan':
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

pca_3dplot_lung(13, 'GTE3')
pca_3dplot_lung(14, 'Gn')
pca_3dplot_lung(15, 'GmE')
pca_3dplot_lung(16, 'GESKtumoryeriiii3kolon')
pca_3dplot_lung(17, 'GESKDiferansiyasyon')
pca_3dplot_lung(18, 'ESKperinörinvdegerikolon')