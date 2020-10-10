# -*- coding: utf-8 -*-
# XGB Regression

from util import *
from xgboost.sklearn import XGBRegressor


# run cross validation for XGB
def run_xgb(random_state=0, poly_degree=1, n_estimators=1000, max_depth=4, learning_rate=0.2,objective="reg:logistic", problem="compressive"):
    print("Running XGB for %s data with random_state=%s, poly_degree=%s, "
          "n_estimators=%s, max_depth=%s, learning_rate=%s, objective=%s" %
          (problem, random_state, poly_degree, n_estimators, max_depth, learning_rate, objective))

    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                         objective=objective, random_state=random_state)
    cv(random_state, poly_degree, model, problem=problem)
    print("Finished running XGB for %s data with random_state=%s, poly_degree=%s, "
          "n_estimators=%s, max_depth=%s, learning_rate=%s, objective=%s\n" %
          (problem, random_state, poly_degree, n_estimators, max_depth, learning_rate, objective))


def random_search():
    options = create_opts()

    # n_estimators
    n_estimators_opts = np.array([100, 200, 500, 1000, 1500, 2000, 2500, 3000, 5000, 10000])  # 10

    # max_depth
    max_depth_opts = np.arange(1, 7, 1)  # 6

    # learning_rate
    learning_rate_opts = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])  # 9

    # objective
    objective_opts = np.array(["reg:linear", "reg:logistic"])  # 2

    for i in range(options.n_run):
        n_estimators = np.random.choice(n_estimators_opts)

        max_depth = np.random.choice(max_depth_opts)

        learning_rate = np.random.choice(learning_rate_opts)

        objective = np.random.choice(objective_opts)

        run_xgb(options.random_state, options.poly_degree, n_estimators, max_depth, learning_rate, objective,
               options.problem)


if __name__ == "__main__":
    # random_search()
    run_xgb(random_state=0, poly_degree=4, n_estimators=2000, max_depth=3, learning_rate=0.1,
           objective="reg:linear", problem="compressive")
