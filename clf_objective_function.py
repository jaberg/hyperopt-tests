import numpy as np
import time
from hyperopt import STATUS_OK


def val_test_error(clf, k_fold, folds, X, y):
    # This will work only if all the algorithms used have
    # fit and score. Will fix when some algorithm doesn't have those.
    validation_error, test_error = 0, 0
    for train, test in k_fold:
        clf.fit(X[train], y[train])
        # test_error += clf.score(X[test], y[test])
        validation_error += clf.score(X[test], y[test])

    validation_error /= folds
    validation_error = -validation_error
    test_error = validation_error
    return validation_error, test_error


# Return algorithm used, so that it's in trials.
# Should change so it can include validation and test input.
def classifier_objective(config, X, y):
    # print config
    folds, begin_time = 10, time.time()
    from sklearn import cross_validation as cv
    k_fold = cv.StratifiedKFold(y, n_folds=folds)

    # shuffle the data
    np.random.seed(0)
    order = np.random.permutation(len(X))
    X, y = X[order], y[order].astype(np.float)
    
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
            
    validation_error, test_error = val_test_error(clf, k_fold, folds,
                                                  X, y)
        
    return {
        'loss': validation_error,
        'status': STATUS_OK,
        'train_time': time.time() - begin_time,
        'true_loss': test_error
    }
