"""
Speed Test for Hyperparameter Optimization

Gaussian Process Example from
https://scikit-optimize.github.io/notebooks/hyperparameter-optimization.html
"""

import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize

boston = load_boston()
X, y = boston.data, boston.target
n_features = X.shape[1]

reg = GradientBoostingRegressor(n_estimators=50, random_state=0)

@profile
def objective(space):
    max_depth, learning_rate, max_features, min_samples_split, min_samples_leaf = space

    reg.set_params(max_depth=max_depth,
                   learning_rate=learning_rate,
                   max_features=max_features,
                   min_samples_split=min_samples_split,
                   min_samples_leaf=min_samples_leaf)

    return -np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=-1,
                                    scoring="neg_mean_absolute_error"))
def main():
    space  = [(1, 5),                           # max_depth
              (10**-5, 10**0, "log-uniform"),   # learning_rate
              (1, n_features),                  # max_features
              (2, 100),                         # min_samples_split
              (1, 100)]                         # min_samples_leaf

    res_gp = gp_minimize(objective, space, n_calls=100, random_state=0, verbose=True)

if __name__=='__main__':
    main()
