from hyperopt import hp
from random import choice
# You can't specify specific arguments inside algorithms if you haven't
# specified the type of algorithm you are going to use.

# In order to specify a parameter, all of the params 'above' it should
# be specified.

# If specified preprocessing algorithm which cannot be used, uses the
# ones that it can (including none) instead of always none.


def check_algo_preprocess(Ualgo, Upreprocess):
    '''
    Returns True if Upreprocess can be used before applying Ualgo
    
    So far, the only blacklisted is naive_bayes - any scalling gives
    ValueError: Input X must be non-negative. naive_bayes scalling
    '''
    black_list_preprocess = {'naive_bayes': ['scale', 'std_scaler'],
                             'svm': [],
                             'dtree': [],
                             'neighbors': []}
    if Upreprocess in black_list_preprocess[Ualgo]:
        return False
    else:
        return True

        
# All the helper_algorithm return space in hp format including every
# subtype of that algorithm.
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
        {'ktype': 'kneighbors', 'n_neighbors': hp.quniform('num', 3,
                                                           19, 1)},
        {'ktype': 'radiusneighbors', 'radius': hp.uniform('rad', 0, 2),
         'out_label': 1}
    ])


def helper_svm():
    return hp.choice('svm_kernel', [
        {'ktype': 'linear'},
        {'ktype': 'rbf', 'width': hp.lognormal('svm_rbf_width', 0, 1)}
    ])


def helper_preprocess(Upreprocess, Unorm, Unaxis, Uw_mean, Uw_std,
                      Usaxis, Ufeature_range, Un_components, Uwhiten):
    '''
    Builds a dictionary containing preprocessing algorithm. The algo
    must be specified.
    '''
    if Upreprocess == 'normalizer':
        return {'palgo': Upreprocess, 'pnorm': Unorm}
    elif Upreprocess == 'normalize':
        return {'palgo': Upreprocess, 'pnorm': Unorm, 'pnaxis': Unaxis}
    elif Upreprocess == 'std_scaler':
        return {'palgo': Upreprocess, 'pw_mean': Uw_mean,
                'pw_std': Uw_std}
    elif Upreprocess == 'scale':
        return {'palgo': Upreprocess, 'pw_mean': Uw_mean,
                'pw_std': Uw_std, 'psaxis': Usaxis}
    elif Upreprocess == 'min_max':
        return {'palgo': Upreprocess, 'pfeature_range': Ufeature_range}
    elif Upreprocess == 'PCA':
        return {'palgo': Upreprocess, 'pn_components': Un_components,
                'pwhiten': Uwhiten}


def get_allowed_pre(Utype, Unorm=choice(['l1', 'l2']), Unaxis=1,
                    Uw_mean=choice([True, False]), Uw_std=True,
                    Usaxis=0, Ufeature_range=(0, 1), Un_components=None,
                    Uwhiten=hp.choice('whiten_choice', [True, False])):
    '''
    Returns a hp space of every preprocessing algorithm that can be
    used combined with Utype algorithm. Utype has to be a string (name)
    '''
    allowed = []
    all_pre = ['scale', 'std_scaler', 'normalize', 'normalizer', 'PCA',
               'min_max']

    for pre_process in all_pre:
        if check_algo_preprocess(Utype, pre_process):
            allowed.append(
                helper_preprocess(
                    pre_process, Unorm, Unaxis, Uw_mean, Uw_std,
                    Usaxis, Ufeature_range, Un_components, Uwhiten))

    allowed.append({'palgo': False})
    return hp.choice('pre_process_algo=' + Utype, allowed)


def get_preprocess(Utype, Upreprocess, Unorm, Unaxis, Uw_mean, Uw_std,
                   Usaxis, Ufeature_range, Un_components, Uwhiten):
    '''
    Utype is specified when we call this function. If Upreprocess is
    also specified, this builds that algo with the params given. If not
    or if that preprocessing algo is black listed, builds hp space with
    all the algos available.
    '''
    if not check_algo_preprocess(Utype, Upreprocess) or Upreprocess is True:
        return get_allowed_pre(Utype)
    elif Upreprocess is False:
        return {'palgo': False}
    else:  # I know which Upreprocess to use
        return helper_preprocess(
            Upreprocess, Unorm, Unaxis, Uw_mean, Uw_std, Usaxis,
            Ufeature_range, Un_components, Uwhiten)


# All the get_algorithm return space as dict. They respect the params
# given the user.
def get_bayes(UNBktype, Ualpha, Ufit_prior, Ubinarize, get_preprocessor):
    preprocess_dict = get_preprocessor('naive_bayes')

    if UNBktype == 'gaussian':
        return {'subtype': {'ktype': 'gaussian'},
                'type': 'naive_bayes',
                'preprocess': preprocess_dict}
    elif UNBktype == 'multinomial':
        return {'subtype': {'ktype': 'multinomial',
                            'alpha': Ualpha,
                            'fit_prior': Ufit_prior},
                'type': 'naive_bayes',
                'preprocess': preprocess_dict}
    elif UNBktype == 'bernoulli':
        return {'subtype': {'ktype': 'bernoulli',
                            'alpha': Ualpha,
                            'fit_prior': Ufit_prior,
                            'binarize': Ubinarize},
                'type': 'naive_bayes',
                'preprocess': preprocess_dict}
    else:
        return {'subtype': UNBktype, 'type': 'naive_bayes',
                'preprocess': preprocess_dict}


def get_svm(UC, USVMktype, Uwidth, get_preprocessor):
    preprocess_dict = get_preprocessor('svm')
    if USVMktype == 'linear':
        return {'kernel': {'ktype': 'linear'}, 'C': UC, 'type': 'svm',
                'preprocess': preprocess_dict}
    elif USVMktype == 'rbf':
        return {'kernel': {'ktype': 'rbf', 'width': Uwidth},
                'C': UC, 'type': 'svm', 'preprocess': preprocess_dict}
    else:
        return {'type': 'svm',
                'C': UC,
                'kernel': USVMktype,
                'preprocess': preprocess_dict}


def get_dtree(Ucriterion, Umax_depth, Umin_samples_split, Upreprocess,
              Unorm, Unaxis, Uw_mean, Uw_std, Usaxis, Ufeature_range,
              Un_components, Uwhiten):
    preprocess_dict = get_preprocess(
        'dtree', Upreprocess, Unorm, Unaxis, Uw_mean, Uw_std,
        Usaxis, Ufeature_range, Un_components, Uwhiten)
    return {'min_samples_split': Umin_samples_split,
            'max_depth': Umax_depth, 'criterion': Ucriterion,
            'type': 'dtree', 'preprocess': preprocess_dict}


def get_neighbors(UNktype, Uweights, Ualgo, Uleaf_sz,
                  Up, Un_neighbors, Uradius, Uout_label, Upreprocess,
                  Unorm, Unaxis, Uw_mean, Uw_std, Usaxis,
                  Ufeature_range, Un_components, Uwhiten):
    preprocess_dict = get_preprocess(
        'neighbors', Upreprocess, Unorm, Unaxis, Uw_mean, Uw_std,
        Usaxis, Ufeature_range, Un_components, Uwhiten)
    if UNktype == 'kneighbors':
        return {'subtype': {'n_neighbors': Un_neighbors,
                            'ktype': UNktype},
                'weights': Uweights, 'algo': Ualgo, 'p': Up,
                'leaf_sz': Uleaf_sz, 'type': 'neighbors',
                'preprocess': preprocess_dict}
    elif UNktype == 'radiusneigbors':
        return {'subtype': {'radius': Uradius,
                            'ktype': UNktype,
                            'outlier_label': Uout_label},
                'weights': Uweights, 'algo': Ualgo, 'p': Up,
                'leaf_sz': Uleaf_sz, 'type': 'neighbors',
                'preprocess': preprocess_dict}
    else:
        return {'subtype': UNktype,
                'weights': Uweights, 'algo': Ualgo, 'p': Up,
                'leaf_sz': Uleaf_sz, 'type': 'neighbors',
                'preprocess': preprocess_dict}
                

# outlier_label defaults to None as in sklearn
def get_space(Utype=True,
              UNBktype=helper_naive_type(),
              Ualpha=hp.lognormal('alpha_', 0, 1),
              Ufit_prior=hp.choice('bool_', [True, False]),
              Ubinarize=hp.choice('binarize_', [.0,
                                    hp.lognormal('threshold_', 0, 1)]),
              UC=hp.lognormal('svm_C', 0, 2),
              Uwidth=hp.lognormal('svm_rbf_width', 0, 1),
              USVMktype=helper_svm(),
              Ucriterion=hp.choice('dtree_criterion', ['entropy',
                                                       'gini']),
              Umax_depth=hp.choice('dtree_max_depth',
                                   [None, 1 + hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
              Umin_samples_split=1 + hp.qlognormal('dtree_min_samples_split', 2, 1, 1),
              Uweights=hp.choice('weighting', ['uniform', 'distance']),
              Ualgo=hp.choice('algos', ['auto', 'brute',
                                        'ball_tree', 'kd_tree']),
              Uleaf_sz=20+hp.randint('size', 20),
              Up=hp.choice('distance', [1, 2]),
              Un_neighbors=hp.quniform('num', 3, 19, 1),
              Uradius=hp.uniform('rad', 0, 2),
              UNktype=helper_neighbors(),
              Uout_label=None,
              Upreprocess=True,
              Unorm=choice(['l1', 'l2']),
              Unaxis=1, Uw_mean=choice([True, False]), Uw_std=True,
              Usaxis=0, Ufeature_range=(0, 1), Un_components=None,
              Uwhiten=hp.choice('whiten_chose', [True, False])):

    def get_preprocessor(name):
         return get_preprocess(name, Upreprocess, Unorm, Unaxis, Uw_mean, Uw_std,
                        Usaxis, Ufeature_range, Un_components, Uwhiten)

    give_me_bayes = get_bayes(UNBktype, Ualpha, Ufit_prior, Ubinarize,
                              get_preprocessor)
    give_me_svm = get_svm(UC, USVMktype, Uwidth, get_preprocessor)

    #TODO: use get_processor
    give_me_dtree = get_dtree(
        Ucriterion, Umax_depth, Umin_samples_split, 
        Unaxis, Uw_mean, Uw_std, Usaxis, Ufeature_range, Un_components,
        Uwhiten)
    give_me_neighbors = get_neighbors(
        UNktype, Uweights, Ualgo, Uleaf_sz, Up, Un_neighbors, Uradius,
        Uout_label, Upreprocess, Unorm, Unaxis, Uw_mean, Uw_std,
        Usaxis, Ufeature_range, Un_components, Uwhiten)

    if Utype == 'naive_bayes':
        res_space = give_me_bayes
    elif Utype == 'svm':
        res_space = give_me_svm
    elif Utype == 'dtree':
        res_space = give_me_dtree
    elif Utype == 'neighbors':
        res_space = give_me_neighbors
    else:
        return hp.choice('quick_fix',
                         [give_me_bayes,
                          give_me_svm,
                          give_me_dtree,
                          give_me_neighbors])

    return hp.choice('quick_fix', [res_space])
