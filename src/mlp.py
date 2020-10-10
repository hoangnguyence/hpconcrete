# -*- coding: utf-8 -*-
# Multilayer Perceptron

from util import *
from sklearn.neural_network import MLPRegressor


# run cross validation for MLP
def run_mlp(random_state=0, poly_degree=3, hd_layer_1=300, hd_layer_2=100, solver="lbfgs", max_iter=1000, alpha=0, problem="compressive"):
    
    print("Running MLP for %s data with "
          "random_state=%s, poly_degree=%s, hd_layer_1=%s, hd_layer_2=%s, solver=%s, max_iter=%s, alpha=%s" %
          (problem, random_state, poly_degree, hd_layer_1, hd_layer_2, solver, max_iter, alpha))
    hd_layers = (hd_layer_1, hd_layer_2, )
    if hd_layer_2 < 0:
        hd_layers = (hd_layer_1, )

    model = MLPRegressor(warm_start=False, random_state=random_state,
                         hidden_layer_sizes=hd_layers, solver=solver, max_iter=max_iter, alpha=alpha)
    
    cv(random_state, poly_degree, model, problem=problem)
    print("Finished running MLP for %s data with "
          "random_state=%s, poly_degree=%s, hd_layer_1=%s, hd_layer_2=%s, solver=%s, max_iter=%s, alpha=%s" %
          (problem, random_state, poly_degree, hd_layer_1, hd_layer_2, solver, max_iter, alpha))


def random_search():
    options = create_opts()

    # hd layer size 1
    hd_layer_1_opts = np.array([100, 200, 300, 400, 500, 1000, 2000])  # 7

    # hd layer size 2
    hd_layer_2_opts = np.array([-1, 100, 200, 300, 400, 500])  # 6

    # max iter
    max_iter_opts = np.array([100, 200, 300, 400, 500, 1000])  # 6

    for i in range(options.n_run):
        hd_layer_1 = np.random.choice(hd_layer_1_opts)

        hd_layer_2 = np.random.choice(hd_layer_2_opts)

        max_iter = np.random.choice(max_iter_opts)
        run_mlp(options.random_state, options.poly_degree, hd_layer_1, hd_layer_2, max_iter, options.problem)


if __name__ == "__main__":
    random_search()
    # run_mlp(poly_degree=1, hd_layer_1=100, hd_layer_2=100, problem="tensile12")
    # run_mlp(poly_degree=2, hd_layer_1=300, hd_layer_2=300, max_iter=200, problem="tensile12")
    # run_mlp(poly_degree=3, hd_layer_1=100, hd_layer_2=300, max_iter=300, problem="tensile12")
    # run_mlp(poly_degree=4, hd_layer_1=300, hd_layer_2=100, max_iter=100, problem="tensile12")
    #
    # run_mlp(poly_degree=1, hd_layer_1=300, hd_layer_2=300, problem="tensile2")
    # run_mlp(poly_degree=2, hd_layer_1=200, hd_layer_2=100, max_iter=400, problem="tensile2")
    # run_mlp(poly_degree=3, hd_layer_1=100, hd_layer_2=200, max_iter=1000, problem="tensile2")
    # run_mlp(poly_degree=4, hd_layer_1=200, hd_layer_2=100, max_iter=400, problem="tensile2")
    #
    # run_mlp(poly_degree=1, hd_layer_1=300, hd_layer_2=200, problem="compressive")
    # run_mlp(poly_degree=2, hd_layer_1=100, hd_layer_2=300, problem="compressive")
    # run_mlp(poly_degree=3, hd_layer_1=300, hd_layer_2=100, problem="compressive")
    # run_mlp(poly_degree=4, hd_layer_1=100, hd_layer_2=300, problem="compressive")
