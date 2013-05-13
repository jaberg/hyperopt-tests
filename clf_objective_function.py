import numpy as np
import time
from hyperopt import STATUS_OK, STATUS_FAIL
from random import choice


# Make validation_error 1 - val_err instead of -val_err. Not doing it
# now, since it's easier to debug when it's like this.
def val_test_error(clf, k_fold, folds, X, y):
    '''
    This will work only if all the algorithms used have fit and score.
    Will fix when some algorithm doesn't have those.
    '''
    validation_error, test_error = 0, 0
    for train, test in k_fold:
        try:
            clf.fit(X[train], y[train])
            # test_error += clf.score(X[test], y[test])
            validation_error += clf.score(X[test], y[test])
        except ValueError:
        #except:  # catches all the errors
            return 0, 0, STATUS_FAIL

    validation_error /= folds
    validation_error = - validation_error
    test_error = validation_error
    return validation_error, test_error, STATUS_OK


# preprocessing changes only X
def preprocess_data(X, Upreprocess, tmp):
    from sklearn import preprocessing as pp
    from sklearn import decomposition as dp
    
    if Upreprocess == 'normalizer':
        return pp.Normalizer(tmp['pnorm']).fit_transform(X)
    elif Upreprocess == 'normalize':
        return pp.normalize(X, tmp['pnorm'], tmp['pnaxis'])
    elif Upreprocess == 'std_scaler':
        return pp.StandardScaler(tmp['pw_mean'],
                                 tmp['pw_std']).fit_transform(X)
    elif Upreprocess == 'scale':
        return pp.scale(X, tmp['psaxis'], tmp['pw_mean'],
                        tmp['pw_std'])
    elif Upreprocess == 'min_max':
        return pp.MinMaxScaler(tmp['pfeature_range']).fit_transform(X)
    elif Upreprocess is False:
        return X
    elif Upreprocess == 'PCA':
        return dp.PCA().fit_transform(X)
    else:
        raise NotImplementedError('Requested preprocessing not implemented')
        

# Should change so it can include validation and test input.
def classifier_objective(config, XX, yy):
    print config  # useful for debugging
    X, y = XX.copy(), yy.copy()
    folds, begin_time = 10, time.time()
    from sklearn import cross_validation as cv
    k_fold = cv.StratifiedKFold(y, n_folds=folds)

    # shuffle the data
    np.random.seed(0)
    order = np.random.permutation(len(X))
    X, y = X[order], y[order].astype(np.float)

    # preprocessing the data
    if config['preprocess'] is not False:
        tmp = config['preprocess']
        X = preprocess_data(X, tmp['palgo'], tmp)
    
    if config['type'] == 'svm':
        from sklearn import svm
        if config['kernel']['ktype'] == 'linear':
            clf = svm.SVC(C=config['C'], kernel='linear')
        else:  # kernel == rbf
            clf = svm.SVC(C=config['C'], kernel='rbf',
                          gamma=config['kernel']['width'])
            
    elif config['type'] == 'dtree':
        from sklearn.tree import DecisionTreeClassifier as dtree
        clf = dtree(criterion=config['criterion'],
                    max_depth=config['max_depth'],
                    min_samples_split=config['min_samples_split'])
        
    elif config['type'] == 'naive_bayes':
        from sklearn import naive_bayes as nb
        tmp = config['subtype']
        if tmp['ktype'] == 'gaussian':
            clf = nb.GaussianNB()
        elif tmp['ktype'] == 'multinomial':
            clf = nb.MultinomialNB(alpha=tmp['alpha'],
                                   fit_prior=tmp['fit_prior'])
        else:  # bernoulli
            clf = nb.BernoulliNB(alpha=tmp['alpha'],
                                 binarize=tmp['binarize'],
                                 fit_prior=tmp['fit_prior'])
            
    elif config['type'] == 'neighbors':
        from sklearn import neighbors
        tmp = config['subtype']
        if tmp['ktype'] == 'kneighbors':
            clf = neighbors.KNeighborsClassifier(
                n_neighbors=tmp['n_neighbors'], weights=config['weights'],
                algorithm=config['algo'], leaf_size=config['leaf_sz'],
                p=config['p'])
        else:  # radius neighbors
            clf = neighbors.RadiusNeighborsClassifier(
                radius=tmp['radius'], weights=config['weights'],
                algorithm=config['algo'], leaf_size=config['leaf_sz'],
                p=config['p'], outlier_label=tmp['out_label'])
            
    validation_error, test_error, status = val_test_error(
        clf, k_fold, folds, X, y)
        
    return {
        'loss': validation_error,
        'status': status,
        'train_time': time.time() - begin_time,
        'true_loss': test_error,
        # will be correct when I change validation_error, or test_error
        'precision': 1 - validation_error,
        'algorithm': config['type'],
    }
