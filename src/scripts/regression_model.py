from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import os
import sys
    



ohe_di = pd.read_csv("/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/data/processed/dataframes/ohe_di.csv")[0:70]
ohe_di_fi = pd.read_csv("/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/data/processed/dataframes/ohe_di_fi.csv")[0:70]
cdt_di_ohe = pd.read_csv("/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/data/processed/dataframes/cdt_di_ohe.csv")[0:70]
cdt_di_fi_ohe = pd.read_csv("/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/data/processed/dataframes/cdt_di_fi_ohe.csv")[0:70]
cdt_di = pd.read_csv("/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/data/processed/dataframes/cdt_di.csv")[0:70]
cdt_di_fi = pd.read_csv("/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/data/processed/dataframes/cdt_di_fi.csv")[0:70]

X = ohe_di

data = pd.read_excel('/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/data/raw/lung_cancer.xlsx')
xlsx = pd.ExcelFile('/Users/utku/Desktop/CE/bioinformatics lab/Lung_Cancer/data/raw/lung_cancer.xlsx')

df1 = pd.read_excel(xlsx, sheet_name='akciğer')
df2 = pd.read_excel(xlsx, sheet_name='kontrol')

emptyLiveMonth = 240
data = data.iloc[:][3:len(data)]
data_live_month = np.where(pd.isna(np.ravel(data.iloc[:,5])), emptyLiveMonth, np.ravel(data.iloc[:,5]))
data_live_month_regression = []
y = data_live_month 


def regression_model_cross_validation(X, y, approach):
    actual_values = []
    predicted_values = []

    # kf = StratifiedKFold(10, random_state=42, shuffle=True)
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(X, y):
        # Split the data into train and test sets for the current fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Fit the regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Store the actual and predicted values for evaluation
        actual_values.extend(y_test)
        predicted_values.extend(y_pred)

    # Calculate the evaluation metrics
    rmse = mean_squared_error(actual_values, predicted_values, squared=False)
    mae = mean_absolute_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    # draw_bar_char(actual_values, predicted_values, approach)
    draw_best_line_plot(actual_values, predicted_values, approach)
    # print(actual_values)
    # print(predicted_values)
    # categories = (categorize_values(actual_values, predicted_values))
    # for category, values in categories.items():
    #     x_values = [value[0] for value in values]
    #     y_values = [value[1] for value in values]
    #     print("X values:", x_values)
    #     print("Y values:", y_values)
    return {"rmse": rmse, "mae": mae, "r2": r2 }

def categorize_values(actual_values, predicted_values):
    categories = {
        '0-18': [],
        '18-36': [],
        '36-96': [],
        '96-120': [],
        '120+': []
    }
    
    for actual, predicted in zip(actual_values, predicted_values):
        if actual < 18:
            categories['0-18'].append((actual, predicted))
        elif actual < 36:
            categories['18-36'].append((actual, predicted))
        elif actual < 96:
            categories['36-96'].append((actual, predicted))
        elif actual < 120:
            categories['96-120'].append((actual, predicted))
        else:
            categories['120+'].append((actual, predicted))
    
    return categories

def draw_best_line_plot(actual_values, predicted_values, approach):
    plt.scatter(actual_values, predicted_values, color='blue', label='Predicted')
    plt.plot(actual_values, actual_values, color='red', label='Actual')

    # Set labels and title
    plt.xlabel('Survival Months')
    plt.ylabel('Predicted Survival Months')
    plt.title("Best Line For " +approach)

    # Add a legend
    save_path = "figures/paper_figures/png/best_line_"+ approach + ".png"
    plt.savefig(save_path)
    # plt.legend()

    # # Show the plot
    # plt.show()

def find_indexes(array):
    indexes = {
        '0-18': [],
        '18-36': [],
        '36-96': [],
        '96-120': [],
        '120+': []
    }
    
    for i, num in enumerate(array):
        if num < 18:
            indexes['0-18'].append(i)
        elif num < 36:
            indexes['18-36'].append(i)
        elif num < 96:
            indexes['36-96'].append(i)
        elif num < 120:
            indexes['96-120'].append(i)
        else:
            indexes['120+'].append(i)
    
    return indexes


# print("0-18:", indexes['0-18'])
# print("18-36:", indexes['18-36'])
# print("36-96:", indexes['36-96'])
# print("96-120:", indexes['96-120'])
# print("120+:", indexes['120+'])

def draw_bar_char(actual_values, predicted_values, approach):
    # Assuming you have calculated RMSE, MAE and R2 for each category
    categories = (categorize_values(actual_values, predicted_values))
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    for category, values in categories.items():
        x_values = [value[0] for value in values]
        y_values = [value[1] for value in values]
        print("X values:", x_values)
        print("Y values:", y_values)
        rmse = mean_squared_error(x_values, y_values, squared=False)
        mae = mean_absolute_error(x_values, y_values)
        r2 = r2_score(x_values, y_values)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append((r2))
    # rmse_scores = [1.2, 0.9, 1.5, 1.8, 2.2]  # Replace with your RMSE scores
    # mae_scores = [0.8, 0.6, 1.1, 1.4, 1.9]  # Replace with your MAE scores
    # r2_scores = [0.85, 0.92, 0.78, 0.72, 0.65]  # Replace with your R2 scores

    # Set the width of the bars
    bar_width = 0.25
    categoriesBarChar = ['0-18', '18-36', '36-96', '96-120', '120+']
    bar_positions = np.arange(len(categoriesBarChar))
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# İlk grafik - RMSE ve MAE
    axs[0].bar(bar_positions, rmse_scores, width=bar_width, label='RMSE')
    axs[0].bar(bar_positions + bar_width, mae_scores, width=bar_width, label='MAE')
    axs[0].set_xticks(bar_positions + bar_width)
    axs[0].set_xticklabels(categoriesBarChar)
    # axs[0].set_ylabel('Scores')
    axs[0].set_title('Approach ' + approach + ' (RMSE and MAE)')
    axs[0].legend()

    # İkinci grafik - R2
    axs[1].bar(bar_positions, r2_scores, width=bar_width, label='R2')
    axs[1].set_xticks(bar_positions + bar_width)
    axs[1].set_xticklabels(categoriesBarChar)
    # axs[1].set_ylabel('Scores')
    axs[1].set_title('Approach ' + approach + ' (R2)')
    axs[1].legend()

    # Grafikleri kaydet
    save_path = "figures/paper_figures/png/model_performance_combined_"+ approach + ".png"
    plt.savefig(save_path)

    # Grafiği göster
    # plt.show()
    

regression_model_cross_validation(ohe_di.values, data_live_month, "OHE_DI")
regression_model_cross_validation(ohe_di_fi.values, data_live_month, "OHE_DI_FI")
regression_model_cross_validation(cdt_di_ohe.values, data_live_month, "CDT_DI_OHE")
regression_model_cross_validation(cdt_di_fi_ohe.values, data_live_month, "CDT_DI_FI_OHE")
regression_model_cross_validation(cdt_di.values, data_live_month, "CDT_DI")
regression_model_cross_validation(cdt_di_fi.values, data_live_month, "CDT_DI_FI")


def regression_model(X, y):
    # data_live_month_control = [emptyLiveMonth for i in range(70)]

    # data_live_month = np.concatenate([data_live_month, data_live_month_control])

    # Verileri eğitim ve test setlerine ayırın
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    folds = 10  # Number of folds
    scores = cross_val_score(model, X, y, cv=folds, scoring='neg_mean_absolute_error')

    # Convert scores to positive values
    scores = -scores

    # Print the mean score and standard deviation
    print("Mean MAE:", scores.mean())
    print("Standard Deviation:", scores.std())
   
    return {"rmse": rmse, "mae": mae, "r2": r2 }
# train_mse = mean_squared_error(y_train, y_train_pred)
# test_mse = mean_squared_error(y_test, y_test_pred)

# print(regression_model_cross_validation(cdt_di_fi.values, data_live_month))
# print(regression_model(cdt_di_fi, data_live_month))