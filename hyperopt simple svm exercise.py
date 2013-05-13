from sklearn import datasets
from hyperopt import fmin, tpe, hp, Trials
import numpy as np
import hyperopt
import space
import clf_objective_function

get_space = space.get_space
classifier_objective = clf_objective_function.classifier_objective

iris = datasets.load_iris()
X, y = iris.data, iris.target
X, y = X[y != 0, :2], y[y != 0]
X_og, y_og = X, y

trials = Trials()
best = fmin(fn=lambda x: classifier_objective(x, X, y),
            space=get_space(Upreprocess='PCA'),
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print '\n\n'
print best
config = hyperopt.space_eval(get_space(Upreprocess='PCA'), best)
print classifier_objective(config, X, y)
# print trials.results
