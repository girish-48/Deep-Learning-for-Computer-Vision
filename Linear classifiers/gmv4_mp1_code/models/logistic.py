"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float, seed: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        # self.w = None  # TODO: change this
        np.random.seed(seed)
        self.w = 0.01*np.random.uniform(-1,1,size=(11,))
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        # Hint: To prevent numerical overflow, try computing the sigmoid for positive numbers and negative numbers separately.
        #       - For negative numbers, try an alternative formulation of the sigmoid function.
        return np.where(z>=0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the logistic regression update rule as introduced in lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1) and scaled by 0.01. 
        - This initialization prevents the weights from starting too large, which can cause saturation of the sigmoid function 

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data; N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        batch_size = 64
        N = X_train.shape[0]
        lr_decay = 0.6
        X_train[:,0] = 1
        epoch = 0

        while epoch < self.epochs : 
            # Data preparation for each epoch
            shuffled_i = np.random.permutation(N)
            X_train = X_train[shuffled_i]
            y_train = y_train[shuffled_i]
            y_relabeled = 2*y_train - 1

            for i in range(0, N, batch_size) : 
                # data processing
                di = min(batch_size, N-i)
                X = X_train[i:i+di]
                y = y_relabeled[i:i+di]

                # updates
                self.w += self.lr*self.sigmoid(-X@self.w*y)*y@X/di
                self.lr *= lr_decay
            epoch += 1

        return


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:exce
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        X_test[:,0] = 1
        z = self.sigmoid(X_test@self.w) 

        return np.where(z>self.threshold, 1, 0)