# This code was adapted from course material by Jenna Wiens (UMichigan).

# python libraries
import os

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

import math
import timeit

######################################################################
# classes
######################################################################

class Data :

    def __init__(self, X=None, y=None) :
        """
        Data class.

        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """

        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y

    def load(self, filename) :
        """
        Load csv file into X array of features and y array of labels.

        Parameters
        --------------------
            filename -- string, filename
        """

        # determine filename
        dir = os.path.dirname('__file__')
        f = os.path.join(dir, '..', 'data', filename)

        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")

        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]

    def plot(self, **kwargs) :
        """Plot data."""

        if 'color' not in kwargs :
            kwargs['color'] = 'b'

        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.show()

# wrapper functions around Data class
def load_data(filename) :
    data = Data()
    data.load(filename)
    return data

def plot_data(X, y, **kwargs) :
    data = Data(X, y)
    data.plot(**kwargs)


class PolynomialRegression() :

    def __init__(self, m=1, reg_param=0) :
        """
        Ordinary least squares regression.

        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
            lambda_ -- float
                       regularization parameter
        """
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param


    def h_w(self, X) :
        w_t = np.transpose(self.coef_)
        n,d = X.shape
        y = np.empty([n,])
        for i in range(n):
            y[i] = np.dot(w_t, X[i])
        return y


    def generate_polynomial_features(self, X) :
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].

        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features

        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """

        n,d = X.shape

        ### ========== TODO : START ========== ###
        # part b: modify to create matrix for simple linear model
        # Phi = np.ones((n,2))
        # #print(Phi)
        # X_flat = X.flatten()
        # Phi[:,1] = X_flat
        # part g: modify to create matrix for polynomial model
        m = self.m_
        Phi = np.ones((n,m+1))
        X_flat = X.flatten()
        for i in range(m + 1):
            power_x = np.power(X_flat, i)
            Phi[:, i] = power_x

        ### ========== TODO : END ========== ###

        return Phi


    def fit_GD(self, X, y, eta=None,
                eps=0, tmax=10000, verbose=False) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            eta     -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes

        Returns
        --------------------
            self    -- an instance of self
        """
        if self.lambda_ != 0 :
            raise Exception("GD with regularization not implemented")

        if verbose :
            plt.subplot(1, 2, 2)
            plt.xlabel('iteration')
            plt.ylabel(r'$J(\theta)$')
            plt.ion()
            plt.show()

        X = self.generate_polynomial_features(X) # map features
        n,d = X.shape
        eta_input = eta
        self.coef_ = np.zeros(d)                 # coefficients
        err_list  = np.zeros((tmax,1))           # errors per iteration

        # GD loop
        for t in range(tmax) :
            ### ========== TODO : START ========== ###
            # part f: update step size
            # change the default eta in the function signature to 'eta=None'
            # and update the line below to your learning rate function
            if eta_input is None :
                #eta = None # change this line
                eta = float(1) / (1 + tmax)
            else :
                eta = eta_input
            ### ========== TODO : END ========== ###

            ### ========== TODO : START ========== ###
            # part d: update theta (self.coef_) using one step of GD
            # hint: you can write simultaneously update all theta using vector math
            y_lol = self.h_w(X)
            grad_0 = 0
            grad_1 = 0
            for i in range(n) :
                grad_0 += ((y_lol[i] - y[i]) * X[i][0])
                grad_1 += ((y_lol[i] - y[i]) * X[i][1])
            self.coef_[0] = self.coef_[0] - 2 * eta * grad_0
            self.coef_[1] = self.coef_[1] - 2 * eta * grad_1

            # track error
            # hint: you cannot use self.predict(...) to make the predictions
            y_pred = y_lol # change this line
            err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n)
            ### ========== TODO : END ========== ###

            # stop?
            if t > 0 and abs(err_list[t] - err_list[t-1]) <= eps :
                break

            # debugging
            if verbose :
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                plt.subplot(1, 2, 1)
                plt.cla()
                plot_data(x, y)
                self.plot_regression()
                plt.subplot(1, 2, 2)
                plt.plot([t+1], [cost], 'bo')
                plt.suptitle('iteration: %d, cost: %f' % (t+1, cost))
                plt.draw()
                plt.pause(0.05) # pause for 0.05 sec

        print ('number of iterations: %d' % (t+1))

        print(self.coef_)
        return self


    def fit(self, X, y, l2regularize = None ) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            l2regularize    -- set to None for no regularization. set to positive double for L2 regularization

        Returns
        --------------------
            self    -- an instance of self
        """

        X = self.generate_polynomial_features(X) # map features

        ### ========== TODO : START ========== ###
        # part e: implement closed-form solution
        # hint: use np.dot(...) and np.linalg.pinv(...)
        #       be sure to update self.coef_ with your solution
        transpose_x = np.transpose(X)
        X_tX = np.dot(transpose_x, X)
        inv = np.linalg.inv(X_tX)
        x_mult = np.dot(inv, transpose_x)
        self.coef_ = np.dot(x_mult, y)
        ### ========== TODO : END ========== ###


    def predict(self, X) :
        """
        Predict output for X.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features

        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None :
            raise Exception("Model not initialized. Perform a fit first.")

        X = self.generate_polynomial_features(X) # map features

        ### ========== TODO : START ========== ###
        # part c: predict y
        y = self.h_w(X)

        ### ========== TODO : END ========== ###

        return y


    def cost(self, X, y) :
        """
        Calculates the objective function.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            cost    -- float, objective J(theta)
        """
        ### ========== TODO : START ========== ###
        # part d: compute J(theta)
        w_t = np.transpose(self.coef_)
        X = self.generate_polynomial_features(X)
        n,d = X.shape
        asdf = np.empty([n,])
        for i in range(n):
            asdf[i] = np.dot(w_t, X[i])

        cost = 0
        for i in range(n):
            var = (asdf[i] - y[i])
            var = var * var
            cost = cost + var
        ### ========== TODO : END ========== ###
        return cost


    def rms_error(self, X, y) :
        """
        Calculates the root mean square error.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            error   -- float, RMSE
        """
        ### ========== TODO : START ========== ###
        # part h: compute RMSE
        error = 0
        j_w = self.cost(X, y)
        n,d = X.shape
        error = math.sqrt(j_w/float(n))
        ### ========== TODO : END ========== ###
        return error


    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs) :
        """Plot regression line."""
        if 'color' not in kwargs :
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs :
            kwargs['linestyle'] = '-'

        X = np.reshape(np.linspace(0,1,n), (n,1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()


######################################################################
# main
######################################################################

def main() :
    # load data
    train_data = load_data('regression_train.csv')
    test_data = load_data('regression_test.csv')



    ### ========== TODO : START ========== ###
    # part a: main code for visualizations
    #print ('Visualizing data...')
    # plot_data(train_data.X, train_data.y)
    # plot_data(test_data.X, test_data.y)

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # parts b-f: main code for linear regression
    print ('Investigating linear regression...')
    model = PolynomialRegression()
    model.coef_ = np.zeros(2)
    cost_thing = model.cost(train_data.X, train_data.y)
    print(cost_thing)

    model.coef_ = np.zeros(2)
    start_time = timeit.default_timer()
    model.fit_GD(train_data.X, train_data.y, eta = 0.0001)
    elapsed = timeit.default_timer() - start_time
    cost_thing = model.cost(train_data.X, train_data.y)
    print("Time: " + str(elapsed))
    print(cost_thing)

    model.coef_ = np.zeros(2)
    start_time = timeit.default_timer()
    model.fit_GD(train_data.X, train_data.y, eta = 0.001)
    elapsed = timeit.default_timer() - start_time
    cost_thing = model.cost(train_data.X, train_data.y)
    print("Time: " + str(elapsed))
    print(cost_thing)

    model.coef_ = np.zeros(2)
    start_time = timeit.default_timer()
    model.fit_GD(train_data.X, train_data.y, eta = 0.01)
    elapsed = timeit.default_timer() - start_time
    cost_thing = model.cost(train_data.X, train_data.y)
    print("Time: " + str(elapsed))
    print(cost_thing)

    model.coef_ = np.zeros(2)
    start_time = timeit.default_timer()
    model.fit_GD(train_data.X, train_data.y, eta = 0.0407)
    elapsed = timeit.default_timer() - start_time
    cost_thing = model.cost(train_data.X, train_data.y)
    print("Time: " + str(elapsed))
    print(cost_thing)

    model.coef_ = np.zeros(2)
    start_time = timeit.default_timer()
    model.fit(train_data.X, train_data.y)
    elapsed = timeit.default_timer() - start_time
    cost_thing = model.cost(train_data.X, train_data.y)
    print("Nut sack")
    print("Time: " + str(elapsed))
    print(cost_thing)

    model.coef_ = np.zeros(2)
    start_time = timeit.default_timer()
    model.fit_GD(train_data.X, train_data.y)
    elapsed = timeit.default_timer() - start_time
    cost_thing = model.cost(train_data.X, train_data.y)
    print("Time: " + str(elapsed))
    print(cost_thing)

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # parts g-i: main code for polynomial regression
    print ('Investigating polynomial regression...')
    train_err_arr = []
    test_err_arr = []
    for i in range(11):
        model_train = PolynomialRegression(m = i)
        model_train.coef_ = np.zeros(i)
        model_train.fit(train_data.X, train_data.y)

        error_thing_train = model_train.rms_error(train_data.X, train_data.y)
        error_thing_test = model_train.rms_error(test_data.X, test_data.y)

        train_err_arr.append(error_thing_train)
        test_err_arr.append(error_thing_test)

    degree = list(range(0,11))

    plt.plot(degree, train_err_arr, label = "training")
    plt.plot(degree, test_err_arr, label = "test")
    plt.legend()
    plt.xlabel("degree")
    plt.ylabel("error")

    # plt.show()
    ### ========== TODO : END ========== ###


    print ("Done!")

if __name__ == "__main__" :
    main()
