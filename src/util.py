# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import *
import time
from optparse import OptionParser
import matplotlib
from matplotlib import pyplot as plt
from matplotlib2tikz import save as tikz_save



def mean_absolute_percentage_error(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100


def create_opts():
    problem = "compressive"
    n_run = 20

    # random_state
    random_state_opts = np.array([0, 1, 2, 42])  # 4
    random_state = np.random.choice(random_state_opts)

    # poly_degree
    poly_degree_opts = np.arange(1, 5, 1)  # 4
    poly_degree = np.random.choice(poly_degree_opts)

    parser = OptionParser()
    parser.add_option("--random_state", dest="random_state", type="int", default=random_state)
    parser.add_option("--poly_degree", dest="poly_degree", type="int", default=poly_degree)
    parser.add_option("--problem", dest="problem", default=problem)
    parser.add_option("--n_run", dest="n_run", type="int", default=n_run)
    (options, args) = parser.parse_args()
    return options


# cross validation
def cv(random_state, poly_degree, model, problem="compressive", print_folds=True):
    interaction_only = False
    if problem.lower() == "compressive":
        data_file = "../data/hpc_compressive_strength.xlsx"
        data = pd.read_excel(data_file, sheet_name='data_1133')
        interaction_only = True

    elif problem.lower() == "tensile12":
        data_file = "../data/hpc_tensile_strength.xlsx"
        data = pd.read_excel(data_file, sheet_name='data_12fts_mean')

    elif problem.lower() == "tensile2":
        data_file = "../data/hpc_tensile_strength.xlsx"
        data = pd.read_excel(data_file, sheet_name='data_2fts')
    else:
        print("The problem has to be compressive or tensile12 or tensile2")
        return

    data = data.values
    n_data_cols = np.shape(data)[1]
    n_features = n_data_cols - 1

    # retrieve data for features
    X = np.array(data[:, :n_features])
    y = np.array(data[:, n_features:])
    # split into 10 folds with shuffle
    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    start_time = time.time()
    scores = []
    fold_index = 0

    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        X_scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

        y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train = y_scaler.fit_transform(y_train)

        if poly_degree >= 1:
            poly = PolynomialFeatures(degree=poly_degree, interaction_only=interaction_only)
            X_train = poly.fit_transform(X_train)
            X_test = poly.transform(X_test)
            # print ('Total number of features: ', X_train.size)

        model.fit(X_train, y_train.ravel())

        y_pred = model.predict(X_test)
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))

        # y_train_pred = model.predict(X_train)
        # y_train_pred = y_scaler.inverse_transform(y_train_pred.reshape(-1, 1))

        # y_train = y_scaler.inverse_transform(y_train)

        # Error measurements
        r_lcc = r2_score(y_test, y_pred) ** 0.5
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        # print("RMSE on train: %s" % mean_squared_error(y_train, y_train_pred) ** 0.5)
        # print("RMSE on test: %s" % rmse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        scores.append([r_lcc, rmse, mae, mape])
        if print_folds:
            print("[fold {0}] r: {1:.5f}, rmse(MPa): {2:.5f}, mae(MPa): {3:.5f}, mape(%): {4:.5f}".
                  format(fold_index, scores[fold_index][0], scores[fold_index][1], scores[fold_index][2], scores[fold_index][3]))
        fold_index += 1
    scores = np.array(scores)
    # barplot(["R2", "RMSE", "MAE", "MAPE"], scores.mean(0), scores.std(0), "Metrics", "Values",
    #         "Performance with different metrics")
    print('k-fold mean:              ', scores.mean(0))
    print('k-fold standard deviation:', scores.std(0))

    # Running time
    print('Running time: %.3fs ' % (time.time() - start_time))
    return scores.mean(0)[1].item()


def barplot(x_data, y_data, error_data, x_label, y_label, title):
    _, ax = plt.subplots()
    x = np.arange(1, len(x_data) + 1)

    ax.bar(x, y_data, color='#539caf', align='center')
    ax.errorbar(x, y_data, yerr=error_data, color='#297083', ls='none', lw=2, capthick=2, elinewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(x_data)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)

    plt.show()


def lineplot(x_data, y_data, label, dashed=True, marker='o', color='blue', markersize=8, linewidth=1.5):
    if dashed:
        plt.plot(x_data, y_data, 'r--', marker=marker, markerfacecolor=color, markersize=markersize, color=color,
                 linewidth=linewidth, label=label)
    else:
        plt.plot(x_data, y_data, marker=marker, markerfacecolor=color, markersize=markersize, color=color,
                 linewidth=linewidth, label=label)


def run_model(regressor, params, random_state=0, poly_degree=1, problem="compressive"):
    model = regressor(**params)
    cv(random_state=random_state, poly_degree=poly_degree, model=model, problem=problem)

def read_data_to_plot(file_path="comparison_data.txt"):
    data = np.loadtxt(file_path, delimiter='	')
    compressive_data = data[:4, :7]
    compressive_x = compressive_data[:, 0]
    mfa_ann = compressive_data[:, 1]
    ann_svr = compressive_data[:, 2]
    svr = compressive_data[:, 3]
    mlp = compressive_data[:, 4]
    gbr = compressive_data[:, 5]
    xgb = compressive_data[:, 6]

    lineplot(compressive_x, ann_svr, "ANN-SVR [C2013]",  marker='', color='black')
    lineplot(compressive_x, mfa_ann, "MFA-ANN [B2018]", dashed=False, marker='', color='black')
    lineplot(compressive_x, svr, "SVR", marker='*', color='blue')
    lineplot(compressive_x, mlp, "MLP", marker='o', color='green')
    lineplot(compressive_x, gbr, "GBR", marker='s', color='cyan')
    lineplot(compressive_x, xgb, "XGB", marker='d', color='red')
    plt.xticks(np.arange(min(compressive_x), max(compressive_x) + 1, 1))
    plt.yticks(np.arange(3.6, 6.4, .2))
    plt.xlabel('Polynomial degree')
    plt.ylabel('RMSE (MPa)')
    plt.legend()
    tikz_save("fig5.tex")
    plt.show()

    tensile_data = data[4:, :]
    tensile_x = tensile_data[:, 0]
    mfa_ann = tensile_data[:, 1]
    svr = tensile_data[:, 2]
    mlp = tensile_data[:, 3]
    gbr = tensile_data[:, 4]
    xgb = tensile_data[:, 5]

    svr12 = tensile_data[:, 6]
    mlp12 = tensile_data[:, 7]
    gbr12 = tensile_data[:, 8]
    xgb12 = tensile_data[:, 9]

    lineplot(tensile_x, mfa_ann, "MFA-ANN (2fts) [B2018]", dashed=False, marker='', color='black')
    lineplot(tensile_x, svr, "SVR (2fts)", marker='*', color='blue')
    lineplot(tensile_x, mlp, "MLP (2fts)", marker='o', color='green')
    lineplot(tensile_x, gbr, "GBR (2fts)", marker='s', color='cyan')
    lineplot(tensile_x, xgb, "XGB (2fts)", marker='d', color='red')

    lineplot(tensile_x, svr12, "SVR (12fts)", marker='*', color='blue')
    lineplot(tensile_x, mlp12, "MLP (12fts)", marker='o', color='green')
    lineplot(tensile_x, gbr12, "GBR (12fts)", marker='s', color='cyan')
    lineplot(tensile_x, xgb12, "XGB (12fts)", marker='d', color='red')

    plt.xticks(np.arange(min(compressive_x), max(compressive_x) + 1, 1))
    plt.yticks(np.arange(0.26, 0.4, .02))
    plt.xlabel('Polynomial degree')
    plt.ylabel('RMSE (MPa)')
    plt.legend()
    tikz_save("fig7.tex")
    plt.show()


if __name__ == "__main__":
    # read_data_to_plot(file_path="comparison_data.txt")
    from sklearn.svm import SVR
    regressor = SVR
    params = {"kernel": "rbf", "C": 500, "epsilon": 0.04, "gamma": 0.1}
    run_model(regressor, params)
