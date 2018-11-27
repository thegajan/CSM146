"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        n,d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        self.probabilities_ = dict()
        majority_val = Counter(y).most_common()
        num = 0
        for item in  majority_val:
            num += item[1]
        for item in majority_val:
            type = item[0]
            self.probabilities_[type] = item[1]/float(num)
        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        prob = [self.probabilities_[0.0], self.probabilities_[1.0]]
        n,d = X.shape
        y = np.random.choice([0.0, 1.0], n, p=prob)

        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2, train_size=0.8) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)

    train_error = []
    test_error = []
    i = 0
    while i < ntrials:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, train_size = train_size, random_state = i)

        clf.fit(X_train, y_train)                  # fit training data using the classifier
        y_pred = clf.predict(X_train)        # take the classifier and run it on the training data
        te = 1 - metrics.accuracy_score(y_train, y_pred, normalize=True)
        train_error.append(te)

        y_pred = clf.predict(X_test)
        tse = 1 - metrics.accuracy_score(y_test, y_pred, normalize=True)
        test_error.append(tse)
        i += 1

    ### ========== TODO : END ========== ###
    train_error = sum(train_error)/len(train_error)
    test_error = sum(test_error)/ len(test_error)
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    
    for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)



    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)



    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    rlf = RandomClassifier() # create MajorityVote classifier, which includes all model parameters
    rlf.fit(X, y)                  # fit training data using the classifier
    y_pred = rlf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print('Classifying using Decision Tree...')
    dtc = DecisionTreeClassifier(criterion = "entropy")
    dtc.fit(X,y)
    y_pred  = dtc.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph

    # save the classifier -- requires GraphViz and pydot
    """
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(dtc, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_pdf("dtree.pdf")
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors
    print('Classifying using k-Nearest Neighbors...')
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X,y)
    y_pred = knn.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')

    print('Classifying using Majority Vote...')
    train_error, test_error = error(clf, X, y)
    print('\t-- training error: %.3f' % train_error)
    print('\t-- testing error: %.3f' % test_error)

    print('Classifying using Random...')
    train_error, test_error = error(rlf, X, y)
    print('\t-- training error: %.3f' % train_error)
    print('\t-- testing error: %.3f' % test_error)

    print('Classifying using Decision Tree...')
    train_error, test_error = error(dtc, X, y)
    print('\t-- training error: %.3f' % train_error)
    print('\t-- testing error: %.3f' % test_error)

    print('Classifying using k-Nearest Neighbors...')
    train_error, test_error = error(knn, X, y)
    print('\t-- training error: %.3f' % train_error)
    print('\t-- testing error: %.3f' % test_error)


    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    i = 1
    index = []
    score = []
    while i < 50:
        k = KNeighborsClassifier(n_neighbors = i)
        cvs = cross_val_score(k, X, y, cv = 10)
        index.append(i)
        score.append(1-sum(cvs)/len(cvs))
        i += 2

    # plt.plot(index, score)
    # plt.xlabel("k")
    # plt.ylabel("error")
    # plt.show()

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    i = 1
    index = []
    training_error = []
    test_error = []
    while i < 20:
        k = DecisionTreeClassifier(criterion = "entropy", max_depth = i)
        tre, tse =  error(k, X, y)
        index.append(i)
        training_error.append(tre)
        test_error.append(tse)
        i += 1

    # plt.plot(index, training_error, label="training error")
    # plt.plot(index, test_error, label="test error")
    # plt.xlabel("depth")
    # plt.ylabel("error")
    # plt.legend(loc='lower left')
    # plt.show()

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    train_split = 0.1
    index = []
    dtc_train = []
    dtc_test = []
    k_train = []
    k_test = []
    while train_split <= 1.0:
        d = DecisionTreeClassifier(criterion = "entropy", max_depth = 4)
        train, test = error(d, X, y, ntrials=100, test_size=0.1, train_size=train_split*0.9)
        dtc_train.append(train)
        dtc_test.append(test)
        k = KNeighborsClassifier(n_neighbors = 7)
        train, test = error(k, X, y, ntrials=100, test_size=0.1, train_size=train_split*0.9)
        k_train.append(train)
        k_test.append(test)
        index.append(train_split)
        train_split += 0.1

    # plt.plot(index, dtc_train, 'b', label="DT Training Error")
    # plt.plot(index, dtc_test, 'c', label="DT Test Error")
    # plt.plot(index, k_train, 'r', label="KNN Training Error")
    # plt.plot(index, k_test, 'm', label="KNN Test Error")
    # plt.xlabel("Training Size")
    # plt.ylabel("Error")
    # plt.legend(loc='lower right')
    # plt.show()

    ### ========== TODO : END ========== ###


    print('Done')


if __name__ == "__main__":
    main()
