"""
Author      : Yi-Chieh Wu, Sriram Sankararman
Description : Twitter
"""

from string import punctuation

import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.

    Parameters
    --------------------
        fname  -- string, filename

    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.

    Parameters
    --------------------
        input_string -- string of characters

    Returns
    --------------------
        words        -- list of lowercase "words"
    """

    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.

    Parameters
    --------------------
        infile    -- string, filename

    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """

    word_list = {}
    list = []
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        for i in fid:
            list = list + extract_words(i)
        for j in list:
            if j not in word_list:
                asdf = len(word_list)
                word_list.update({j:asdf})

        ### ========== TODO : END ========== ###

    return word_list

def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.

    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)

    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """

    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))

    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        for i, line in enumerate(fid, 0):
            asdf = extract_words(line)
            for j in asdf:
                if j in word_list:
                    x = word_list[j]
                    feature_matrix[i][x] = 1
        ### ========== TODO : END ========== ###

    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the
    true labels and the predicted labels.

    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc'

    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1

    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    if metric == 'accuracy':
        score = metrics.accuracy_score(y_true, y_pred)
    if metric == 'f1_score':
        score = metrics.f1_score(y_true, y_pred)
    if metric == 'auroc':
        score = metrics.roc_auc_score(y_true, y_pred)
    return score
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.

    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """

    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance
    t_perf = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        tp = performance(y_test, y_pred, metric)
        t_perf.append(tp)
    t_perf_val = sum(t_perf)/ len(t_perf)
    return t_perf_val
    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.

    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """

    print 'Linear SVM Hyperparameter Selection based on ' + str(metric) + ':'
    C_range = 10.0 ** np.arange(-3, 3)

    ### ========== TODO : START ========== ###
    # part 2: select optimal hyperparameter using cross-validation
    max_c = 0
    max_cvs = 0
    for c in C_range:
        clf = SVC(kernel='linear', C=c)
        cvs = cv_performance(clf, X, y, kf, metric)
        print str(c) + ': ' + '%.4f' % cvs
        if cvs > max_cvs:
            max_cvs = cvs
            max_c = c
    return max_c
    ### ========== TODO : END ========== ###



def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.

    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure

    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 3: return performance on test data by first computing predictions and then calling performance

    score = 0

    y_pred = clf.predict(X)
    score = performance(y, y_pred, metric)

    return score
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################

def main() :
    np.random.seed(1234)

    # read the tweets and its labels
    dictionary = extract_dictionary('../data/tweets.txt')
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    y = read_vector_file('../data/labels.txt')

    metric_list = ["accuracy", "f1_score", "auroc"]

    ### ========== TODO : START ========== ###
    # part 1: split data into training (training + cross-validation) and testing set
    s1 = slice(0, 560, 1)
    s2 = slice(560, 630, 1)
    training_data = X[s1]
    test_data = X[s2]

    training_label = y[s1]
    test_label = y[s2]

    # part 2: create stratified folds (5-fold CV)
    kf = StratifiedKFold(training_label, n_folds=5)
    # part 2: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    for metric in metric_list:
        val = select_param_linear(training_data, training_label, kf, metric)
        print("max c: " + str(val))

    # part 3: train linear-kernel SVMs with selected hyperparameters
    Model = SVC(kernel='linear', C=10)
    Model.fit(training_data, training_label)
    # part 3: report performance on test data
    for metric in metric_list:
        perf = performance_test(Model, test_data, test_label, metric)
        print 'Performance of ' + str(metric) + ': ' + str(perf)

    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
