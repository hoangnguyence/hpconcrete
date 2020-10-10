from svr import *
from mlp import *
from gbr import *
from xgb import *
from util import *

# ------------------------------
# CONCRETE COMPRESSIVE STRENGTH
# ------------------------------

run_svr(random_state=0, poly_degree=1, kernel="rbf", C=1000,
        epsilon=0.04, gamma=0.5, problem="compressive")

# run_mlp(random_state=0, poly_degree=3, hd_layer_1=300,
#         hd_layer_2=100, max_iter=1000, alpha=0, problem="compressive")

# run_gbr(random_state=0, poly_degree=1, n_estimators=1000,
#         max_depth=5, learning_rate=0.1, loss="huber", min_samples_split=6,problem="compressive")

# run_xgb(random_state=0, poly_degree=1, n_estimators=1000, max_depth=4,
#         learning_rate=0.2, objective="reg:logistic", problem="compressive")


# --------------------------
# CONCRETE TENSILE STRENGTH
# --------------------------

# problem="tensile12" (12 features) or problem="tensile2" (2 features)

# run_svr(random_state=0, poly_degree=1, kernel="rbf", C=20,
#         epsilon=0.01, gamma=0.9, problem="tensile12")

# run_mlp(random_state=0, poly_degree=2, hd_layer_1=300,
#         hd_layer_2=300, max_iter=200, alpha=0.0001, problem="tensile12")

# run_gbr(random_state=0, poly_degree=3, n_estimators=1000,
#         max_depth=2, learning_rate=0.1, loss="huber", min_samples_split=6, problem="tensile12")

# run_xgb(random_state=0, poly_degree=2, n_estimators=1000, max_depth=4,
#         learning_rate=0.1, objective="reg:logistic", problem="tensile12")
