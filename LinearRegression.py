#!coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.optimize #fmin_cg to train the linear regression

import warnings
warnings.filterwarnings('ignore')


class LinearRegerssion():
    def __int__(self):
        self.iterations = 1500
        self.alpha = 0.01

    def featureNormalize(self, myX):
        """
        Takes as input the X array (with bias "1" first column), does
        feature normalizing on the columns (subtract mean, divide by standard deviation).
        Returns the feature-normalized X, and feature means and stds in a list
        Note this is different than my implementation in assignment 1...
        I didn't realize you should subtract the means, THEN compute std of the
        mean-subtracted columns.
        Doesn't make a huge difference, I've found
        """

        Xnorm = myX.copy()
        stored_feature_means = np.mean(Xnorm, axis=0)  # column-by-column
        Xnorm[:, 1:] = Xnorm[:, 1:] - stored_feature_means[1:]
        stored_feature_stds = np.std(Xnorm, axis=0, ddof=1)
        Xnorm[:, 1:] = Xnorm[:, 1:] / stored_feature_stds[1:]
        return Xnorm, stored_feature_means, stored_feature_stds

    def h(self, theta, X):  # Linear hypothesis function
        return np.dot(X, theta)

    def computeCost(self, mytheta, myX, myy, mylambda=0.):  # Cost function
        """
        theta_start is an n- dimensional vector of initial theta guess
        X is matrix with n- columns and m- rows
        y is a matrix with m- rows and 1 column
        """
        m = myX.shape[0]
        myh = self.h(mytheta, myX).reshape((m, 1))
        mycost = float((1. / (2 * m)) * np.dot((myh - myy).T, (myh - myy)))
        regterm = (float(mylambda) / (2 * m)) * float(mytheta[1:].T.dot(mytheta[1:]))
        return mycost + regterm

    def computeGradient(self, mytheta, myX, myy, mylambda=0.):
        mytheta = mytheta.reshape((mytheta.shape[0], 1))
        m = myX.shape[0]
        # grad has same shape as myTheta (2x1)
        # myh = self.h(mytheta, myX).reshape((m, 1))
        grad = (1. / float(m)) * myX.T.dot(self.h(mytheta, myX) - myy)
        regterm = (float(mylambda) / m) * mytheta
        regterm[0] = 0  # don't regulate bias term
        regterm.reshape((grad.shape[0], 1))
        return grad + regterm

    # This is for the minimization routine that wants everything flattened
    def computeGradientFlattened(self, mytheta, myX, myy, mylambda=0.):
        return self.computeGradient(mytheta, myX, myy, mylambda=0.).flatten()

    def optimizeTheta(self, myTheta_initial, myX, myy, mylambda=0., print_output=True):
        fit_theta = scipy.optimize.fmin_cg(self.computeCost, x0=myTheta_initial, \
                                           fprime=self.computeGradientFlattened, \
                                           args=(myX, myy, mylambda), \
                                           disp=print_output, \
                                           epsilon=1.49e-12, \
                                           maxiter=1000)
        fit_theta = fit_theta.reshape((myTheta_initial.shape[0], 1))
        return fit_theta

    def plotLearningCurve(self, X, y, Xval, yval):
        """
        Loop over first training point, then first 2 training points, then first 3 ...
        and use each training-set-subset to find trained parameters.
        With those parameters, compute the cost on that subset (Jtrain)
        remembering that for Jtrain, lambda = 0 (even if you are using regularization).
        Then, use the trained parameters to compute Jval on the entire validation set
        again forcing lambda = 0 even if using regularization.
        Store the computed errors, error_train and error_val and plot them.
        """
        initial_theta = np.array([[1.], [1.]])
        mym, error_train, error_val = [], [], []
        for x in xrange(1, 13, 1):
            train_subset = X[:x, :]
            y_subset = y[:x]
            mym.append(y_subset.shape[0])
            fit_theta = self.optimizeTheta(initial_theta, train_subset, y_subset, mylambda=0., print_output=False)
            error_train.append(self.computeCost(fit_theta, train_subset, y_subset, mylambda=0.))
            error_val.append(self.computeCost(fit_theta, Xval, yval, mylambda=0.))

        plt.figure(figsize=(8, 5))
        plt.plot(mym, error_train, label='Train')
        plt.plot(mym, error_val, label='Cross Validation')
        plt.legend()
        plt.title('Learning curve for linear regression')
        plt.xlabel('Number of training examples')
        plt.ylabel('Error')
        plt.grid(True)

    def standRegres(self, X, y):
        xT = X.T * X
        if np.linalg.det(xT) == 0.0:
            print "This matrix is singular, cannot do inverse."
            return
        theat = xT.I * (X * y)
        return theat

if __name__ == '__main__':
    datafile = 'data/ex5data1.mat'
    mat = scipy.io.loadmat(datafile)
    # Training set
    X, y = mat['X'], mat['y']
    # Cross validation set
    Xval, yval = mat['Xval'], mat['yval']
    # Test set
    Xtest, ytest = mat['Xtest'], mat['ytest']
    # Insert a column of 1's to all of the X's, as usual
    X = np.insert(X, 0, 1, axis=1)
    Xval = np.insert(Xval, 0, 1, axis=1)
    Xtest = np.insert(Xtest, 0, 1, axis=1)
    lr = LinearRegerssion()
    mytheata = np.array([[1.],[1.]])
    fit_theata = lr.optimizeTheta(mytheata, X, y, 0.)
