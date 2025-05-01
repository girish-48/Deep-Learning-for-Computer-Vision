"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int, seed: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.seed = seed
        self.W = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the perceptron update rule as introduced in the Lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        np.random.seed(self.seed)

        batch_size = 512
        N = X_train.shape[0]
        p = X_train.shape[1]
        lr_decay = 0.6
        # X_train[:,0] = 1
        self.W = 0.01*np.random.uniform(-1,1,size=(self.n_class, p+1)) 
        X_train = np.hstack((X_train,np.ones((N,1))))
        epoch = 0
        M = np.zeros((self.n_class))

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
                accu_gradient = np.zeros((self.n_class, p+1))

                # updates
                WX = self.W@np.transpose(X) 
                for j in range(di) :
                    WX[:,j] -= WX[y[j],j]
                    for n in range(self.n_class) :
                        if n == y[j] : 
                            M[n] = np.sum(WX[:,j]>0)
                        else :
                            M[n] = -int(WX[n,j]>0)

                    accu_gradient += np.outer(M,X[j,:])

                self.W += self.lr*accu_gradient/di
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
