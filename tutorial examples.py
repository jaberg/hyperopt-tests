import numpy as np
from sklearn.linear_model.sgd_fast import SquaredLoss
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL
from hyperopt import Trials
import hyperopt.pyll.stochastic
import pickle
import time


def objective(x):
    return {'loss': x ** 2 + 300,
            'status': STATUS_FAIL if x > 0 else STATUS_OK,
            'eval_time': time.time(),
            'random_stuff': {'something': 'string?'},
            'attachments': {'time_module': pickle.dumps(time.time)}
            }

    
# even if STATUS_FAIL, loss function needed
def objective1(x):
    return {'status': STATUS_FAIL}


# why does hp.randint('x', 10) always return same number?
# works without space=____
trials = Trials()
best = fmin(objective,
            space=hp.choice('a',
                [
                    (1 + hp.randint('c1', 10)),
                    (hp.uniform('c2', -10, 10))
                ]),
            # space=hp.quniform('x', -10, 10, .00001),
            algo=tpe.suggest,
            max_evals=10,
            trials=trials)

# why is this red? where is the syntax error?
print best
# always prints all the floats, regardless of status
print trials.losses()
print trials.statuses()
print trials.results
# here it's not red..
print best

# msg = trials.trial_attachments(trials.trials[5])['time_module']
# time_module = pickle.loads(msg)

# print time_module
# print msg

space = hp.choice('a',
        [
            ('case 1', 1 + hp.randint('c1', 10)),
            ('case 2', hp.uniform('c2', -10, 10))
        ]),
# gives a (random?) point in space?
print hyperopt.pyll.stochastic.sample(space)

space = hp.choice('classifier_type', [
    {
        'type': 'svm',
        'C': hp.lognormal('svm_C', 0, 1),
        'kernel': hp.choice('svm_kernel', [
            {'ktype': 'linear'},
            {'ktype': 'RBF', 'width': hp.lognormal('svm_rbf_width', 0, 1)},
        ]),
    },
    {
        'type': 'dtree',
        'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
        'max_depth': hp.choice('dtree_max_depth',
                               [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
        'min_samples_split': hp.qlognormal('dtree_min_samples_split', 2, 1, 1),
    },
    ])

print hyperopt.pyll.stochastic.sample(space)
# how does this train? how to use it?

best = fmin(SquaredLoss(),
            space=space,
            algo=tpe.suggest,
            max_evals=10)

print best
