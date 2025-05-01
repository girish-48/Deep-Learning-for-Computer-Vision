"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, seed: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.seed = seed
        self.w = None 
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X: np.ndarray, y: np.ndarray, p: int, di : int) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        M = np.zeros((self.n_class))
        accu_gradient = np.zeros((self.n_class, p+1))
        WX = self.W@np.transpose(X) 

        for j in range(di) :
            # WX[:,j] -= WX[y[j],j]-1
            WX[:,j] = WX[:,j] - WX[y[j],j] + 1
            for n in range(self.n_class) :
                if n == y[j] : 
                    # Ignoring correct class, because for correct class always WX = 1
                    M[n] = -(np.sum(WX[:,j]>0)-1)
                else :
                    M[n] = int(WX[n,j]>0)

            accu_gradient += np.outer(M,X[j,:])
        return (self.reg_const*self.W + accu_gradient)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        np.random.seed(self.seed)

        batch_size = 1024 
        N = X_train.shape[0]
        p = X_train.shape[1]
        lr_decay = 0.8
        # X_train[:,0] = 1
        self.W = 0.01*np.random.uniform(-1,1,size=(self.n_class, p+1)) 
        X_train = np.hstack((X_train,np.ones((N,1))))
        epoch = 0

        while epoch < self.epochs : 
            # Data preparation for each epoch
            shuffled_i = np.random.permutation(N)
            X_train = X_train[shuffled_i]
            y_train = y_train[shuffled_i]

            for i in range(0, N, batch_size) : 
                # data processing
                di = min(batch_size, N-i)
                X = X_train[i:i+di]
                y = y_train[i:i+di]
                self.W -= self.lr*(self.calc_gradient(X,y,p,di))/di
            self.lr *= lr_decay
            epoch += 1

        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        X_test = np.hstack((X_test,np.ones((X_test.shape[0],1))))
        Y = np.argmax(self.W@np.transpose(X_test),axis=0)
        return Y
