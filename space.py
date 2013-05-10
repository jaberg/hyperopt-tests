from hyperopt import hp


space = hp.choice('classifier_type', [
    {
        'type': 'naive_bayes',
        # threshold for binarize, how to get a 'meaningful' number?
        'subtype': hp.choice('naive_subtype', [
            {'ktype': 'gaussian'},
            {'ktype': 'multinomial', 'alpha': hp.lognormal('alpha_mult', 0, 1),
             'fit_prior': hp.choice('bool_mult', [False, True])},
            {'ktype': 'bernoulli', 'alpha': hp.lognormal('alpha_ber', 0, 1),
             'fit_prior': hp.choice('bool_ber', [False, True]),
             'binarize': hp.choice('binarize_or_not',
                                   [
                                       .0,
                                       hp.lognormal('threshold', 0, 1)
                                   ])}
        ])
    },
    {
        'type': 'svm',
        'C': hp.lognormal('svm_C', 0, 1),
        'kernel': hp.choice('svm_kernel', [
           {'ktype': 'linear'},
            {'ktype': 'RBF', 'width': hp.lognormal('svm_rbf_width', 0, 1)}
        ])
    },
    {
        'type': 'dtree',
        'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
        'max_depth': hp.choice('dtree_max_depth',
                [None, 1 + hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
        'min_samples_split': 1 + hp.qlognormal('dtree_min_samples_split', 2, 1, 1),
    },
    {
        # modify for weights callable (user defined)
        # not using p > 2 (minkowski_distance l_p)
        # by default there is outlier_label
        # there is a problem with radius neighbors, which I can't figure
        # out... so for now, it's not using it
        'type': 'neighbors',
        'weights': hp.choice('weighting', ['uniform', 'distance']),
        'algo': hp.choice('algos', ['auto', 'brute', 'ball_tree', 'kd_tree']),
        'leaf_sz': 20 + hp.randint('size', 20),
        'p': hp.choice('distance', [1, 2]),
        'subtype': hp.choice('neighbor_type', [
            {'ktype': 'kneighbors', 'n_neighbors': hp.quniform('num', 3, 19, 1)},
            #{'ktype': 'radiusneighbors', 'radius': hp.uniform('rad', 0, 2),
            #'out_label': 1}
        ])
    },
])
