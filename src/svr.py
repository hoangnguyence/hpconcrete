# -*- coding: utf-8 -*-
# SVM Regression

from util import *
from sklearn.svm import SVR


# run cross validation for SVR
def run_svr(random_state=0, poly_degree=1, kernel="rbf", C=1000, epsilon=0.04, gamma=0.5, problem="compressive"):
    print("Running SVR for %s data with "
          "random_state=%s, poly_degree=%s, kernel=%s, C=%s, epsilon=%s, gamma=%s" %
          (problem, random_state, poly_degree, kernel, C, epsilon, gamma))

    model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, degree=poly_degree)
    if kernel == "poly":
        poly_degree = 1

    cv(random_state, poly_degree, model, problem=problem)
    print("Finished running SVR for %s data with "
          "random_state=%s, poly_degree=%s, kernel=%s, C=%s, epsilon=%s, gamma=%s\n" %
          (problem, random_state, poly_degree, kernel, C, epsilon, gamma))


def random_search():
    options = create_opts()

    # kernel
    kernel_opts = np.array(["rbf", "rbf", "poly", "linear"])  # 3

    # C
    C_opts = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]) # 15
    C_linear_opts = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]) # 10

    # epsilon
    epsilon_opts = np.array([.01, .02, .03, .04, .05, .06, .07, .08, .09, 0.1])  # 10

    # gamma
    gamma_opts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # 9

    for i in range(options.n_run):
        kernel = str(np.random.choice(kernel_opts))

        C = np.random.choice(C_opts)
        if kernel == "linear":
            C = np.random.choice(C_linear_opts)

        epsilon = np.random.choice(epsilon_opts)

        gamma = np.random.choice(gamma_opts)

        run_svr(options.random_state, options.poly_degree, kernel, C, epsilon, gamma, options.problem)


if __name__ == "__main__":
    random_search()