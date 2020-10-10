# -*- coding: utf-8 -*-
# Gradient Boosting Regression

from util import *
from sklearn.ensemble import GradientBoostingRegressor


# run cross validation for GBR
def run_gbr(random_state=0, poly_degree=1, n_estimators=1000,
           max_depth=5, learning_rate=0.1, loss="huber", min_samples_split=6, problem="compressive"):
    print("Running GBR for %s data with random_state=%s, poly_degree=%s, "
          "n_estimators=%s, max_depth=%s, learning_rate=%s, loss=%s, min_samples_split=%s" %
          (problem, random_state, poly_degree, n_estimators, max_depth, learning_rate, loss, min_samples_split))

    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                      loss=loss, random_state=random_state, min_samples_split=min_samples_split)
    cv(random_state, poly_degree, model, problem=problem)
    print("Finished running GBR for %s data with random_state=%s, poly_degree=%s, "
          "n_estimators=%s, max_depth=%s, learning_rate=%s, loss=%s, min_samples_split=%s\n" %
          (problem, random_state, poly_degree, n_estimators, max_depth, learning_rate, loss, min_samples_split))


def random_search():
    options = create_opts()

    # n_estimators
    n_estimators_opts = np.array([100, 200, 500, 1000, 1500, 2000, 2500, 3000, 5000, 10000])  # 10

    # max_depth
    max_depth_opts = np.arange(1, 7, 1) # 6

    # learning_rate
    learning_rate_opts = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]) # 9

    # loss
    loss_opts = np.array(["ls", "lad", "huber"]) # 3

    min_samples_split_opts = np.array([2, 3, 4, 5, 6]) # 5

    for i in range(options.n_run):
        n_estimators = np.random.choice(n_estimators_opts)

        max_depth = np.random.choice(max_depth_opts)

        learning_rate = np.random.choice(learning_rate_opts)

        loss = np.random.choice(loss_opts)

        min_samples_split = np.random.choice(min_samples_split_opts)

        run_gbr(options.random_state, options.poly_degree, n_estimators, max_depth, learning_rate, loss,
               min_samples_split, options.problem)


if __name__ == "__main__":
    random_search()
