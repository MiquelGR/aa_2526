import numpy as np


class Perceptron:

    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting. w[0] = threshold
    errors_ : list
        Number of miss classifications in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None  # defined in method fit

    def fit(self, X, y):

        """Fit training dat.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        """
        self.w_ = np.zeros(1 + X.shape[1])  # First position corresponds to threshold
        
        w_j = self.w_[1:] #All weights less the bias
        bias = self.w_[0] #BIAS


        for _ in range(self.n_iter):
            y_pred = self.predict(X)
            error = y - y_pred   # vector de errores

            # actualización global (suma de todas las correcciones)
            w_j =w_j + self.eta * np.dot(error, X) #W_j
            bias = bias + self.eta * np.sum(error) #Bias
            self.w_ = np.concatenate(([bias], w_j)) #Concatenar

    

            


    def predict(self, X):
        """Return class label.
            First calculate the output: (X * weights) + threshold
            Second apply the step function
            Return a list with classes
        """

        w_i = self.w_[1:] #All weight less the bias
        bias = self.w_[0]

        #Calculate output
        z = np.dot(X, w_i) + bias 

        #Prediction
        y_pred = [self.escalon(valor) for valor in z]

        return y_pred  
    
    def escalon(self, x):
        return 1 if x >= 0 else -1
