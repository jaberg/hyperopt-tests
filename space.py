from hyperopt import hp
# You can't specify specific arguments inside algorithms if you haven't
# specified the type of algorithm you are going to use.

# In order to specify a parameter, all of the params 'above' it should
# be specified.

# Radius neighbors not tested and commented. Uncomment to use it.


def get_whole_space():
    return hp.choice('classifier_type', [
        {
            'type': 'naive_bayes',
            # threshold for binarize, how to get a 'meaningful' number?
            'subtype': hp.choice('naive_subtype', [
                {'ktype': 'gaussian'},
                {'ktype': 'multinomial',
                 'alpha': hp.lognormal('alpha_mult', 0, 1),
                 'fit_prior': hp.choice('bool_mult', [False, True])},
                {'ktype': 'bernoulli',
                 'alpha': hp.lognormal('alpha_ber', 0, 1),
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
            'C': hp.lognormal('svm_C', 0, 10),
            'kernel': hp.choice('svm_kernel', [
                {'ktype': 'linear'},
                {'ktype': 'rbf', 'width': hp.lognormal('svm_rbf_width', 0, 1)}
            ])
        },
        {
            'type': 'dtree',
            'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
            'max_depth':
            hp.choice('dtree_max_depth',
                      [None, 1 +
                       hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
            'min_samples_split':
            1 + hp.qlognormal('dtree_min_samples_split', 2, 1, 1),
        },
        {
            # modify for weights callable (user defined)
            # not using p > 2 (minkowski_distance l_p)
            'type': 'neighbors',
            'weights': hp.choice('weighting', ['uniform', 'distance']),
            'algo': hp.choice('algos', ['auto', 'brute',
                                        'ball_tree', 'kd_tree']),
            'leaf_sz': 20 + hp.randint('size', 20),
            'p': hp.choice('distance', [1, 2]),
            'subtype': hp.choice('neighbor_type', [
                {'ktype': 'kneighbors',
                 'n_neighbors': hp.quniform('num', 3, 19, 1)},
                #{'ktype': 'radiusneighbors',
                #'radius': hp.uniform('rad', 0, 2)},
            ])
        },
    ])


def helper_naive_type():
    return hp.choice('naive_subtype', [
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


def helper_neighbors():
    return hp.choice('neighbor_type', [
        {'ktype': 'kneighbors', 'n_neighbors': hp.quniform('num', 3, 19, 1)},
        #{'ktype': 'radiusneighbors', 'radius': hp.uniform('rad', 0, 2),
        #'out_label': 1}
    ])


def helper_svm():
    return hp.choice('svm_kernel', [
        {'ktype': 'linear'},
        {'ktype': 'rbf', 'width': hp.lognormal('svm_rbf_width', 0, 1)}
    ])


def get_bayes(UNBktype, Ualpha, Ufit_prior, Ubinarize):
    if UNBktype == 'gaussian':
        return {'subtype': {'ktype': 'gaussian'}, 'type': 'naive_bayes'}
    elif UNBktype == 'multinomial':
        return {'subtype': {'ktype': 'multinomial',
                            'alpha': Ualpha,
                            'fit_prior': Ufit_prior},
                'type': 'naive_bayes'}
    elif UNBktype == 'bernoulli':
        return {'subtype': {'ktype': 'bernoulli',
                            'alpha': Ualpha,
                            'fit_prior': Ufit_prior,
                            'binarize': Ubinarize},
                'type': 'naive_bayes'}
    else:
        return {'subtype': UNBktype, 'type': 'naive_bayes'}


def get_svm(UC, USVMktype, Uwidth):
    if USVMktype == 'linear':
        return {'kernel': {'ktype': 'linear'}, 'C': UC, 'type': 'svm'}
    elif USVMktype == 'rbf':
        return {'kernel': {'ktype': 'rbf', 'width': Uwidth},
                'C': UC, 'type': 'svm'}
    else:
        return {'type': 'svm',
                'C': UC,
                'kernel': USVMktype}


def get_dtree(Ucriterion, Umax_depth, Umin_samples_split):
    return {'min_samples_split': Umin_samples_split,
            'max_depth': Umax_depth, 'criterion': Ucriterion,
            'type': 'dtree'}


def get_neighbors(UNktype, Uweights, Ualgo, Uleaf_sz,
                  Up, Un_neighbors, Uradius, Uout_label):
    if UNktype == 'kneighbors':
        return {'subtype': {'n_neighbors': Un_neighbors,
                            'ktype': UNktype},
                'weights': Uweights, 'algo': Ualgo, 'p': Up,
                'leaf_sz': Uleaf_sz, 'type': 'neighbors'}
    elif UNktype == 'radiusneigbors':  # (not working)
        return {'subtype': {'radius': Uradius,
                            'ktype': UNktype,
                            'outlier_label': Uout_label},
                'weights': Uweights, 'algo': Ualgo, 'p': Up,
                'leaf_sz': Uleaf_sz, 'type': 'neighbors'}
    else:
        return {'subtype': UNktype,
                'weights': Uweights, 'algo': Ualgo, 'p': Up,
                'leaf_sz': Uleaf_sz, 'type': 'neighbors'}
                

# change UNktype to include radius neighbors
# outlier_label defaults to None as in sklearn
def get_space(Utype=get_whole_space(),
              UNBktype=helper_naive_type(),
              Ualpha=hp.lognormal('alpha_', 0, 1),
              Ufit_prior=hp.choice('bool_', [True, False]),
              Ubinarize=hp.choice('binarize_', [.0,
                                    hp.lognormal('threshold_', 0, 1)]),
              UC=hp.lognormal('svm_C', 0, 10),
              Uwidth=hp.lognormal('svm_rbf_width', 0, 1),
              USVMktype=helper_svm(),
              Ucriterion=hp.choice('dtree_criterion', ['entropy',
                                                       'gini']),
              Umax_depth=hp.choice('dtree_max_depth',
                                   [None, 1 + hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
              Umin_samples_split=1+hp.qlognormal('dtree_min_samples_split', 2, 1,1 ),
              Uweights=hp.choice('weighting', ['uniform', 'distance']),
              Ualgo=hp.choice('algos', ['auto', 'brute',
                                        'ball_tree', 'kd_tree']),
              Uleaf_sz=20+hp.randint('size', 20),
              Up=hp.choice('distance', [1, 2]),
              Un_neighbors=hp.quniform('num', 3, 19, 1),
              Uradius=hp.uniform('rad', 0, 2),
              UNktype=helper_neighbors(),
              Uout_label=None):

    if Utype == 'naive_bayes':
        res_space = get_bayes(UNBktype, Ualpha, Ufit_prior, Ubinarize)
    elif Utype == 'svm':
        res_space = get_svm(UC, USVMktype, Uwidth)
    elif Utype == 'dtree':
        res_space = get_dtree(Ucriterion, Umax_depth,
                              Umin_samples_split)
    elif Utype == 'neighbors':
        res_space = get_neighbors(UNktype, Uweights, Ualgo, Uleaf_sz,
                                  Up, Un_neighbors, Uradius, Uout_label)
    else:
        res_space = Utype

    return hp.choice('quick_fix', [res_space])
