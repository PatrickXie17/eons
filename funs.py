import numpy as np
import scipy
from sklearn.linear_model import LinearRegression

class Eons():
    def __init__(self):
        self

    def l2_loss():
        """
        This function calculate the L2 loss of two arrays
        :param d1: The first np array
        :param d2: The second np array
        :return: MSE
        """
        def tf(d1, d2):
            return np.mean((np.array(d1) - np.array(d2)) ** 2)

        return tf

    def feed_forward_regression(self, weight, *args):
        """
        This is the feed forward step of a neural network
        :param X: Input data. Each column is a variable and each row is an observation.
        :param y: Target.
        :param weight: Weight of each variable.
        :param loss_fun: Loss metric to evaluate the loss.
        :return: Total forward loss.
        """
        X_train = args[0]
        X_test = args[1]
        y_train = args[2]
        y_test = args[3]
        loss_fun = args[4]
        # self-normalization for the weights
        weight = weight / np.sqrt(np.sum(weight ** 2))

        # projection onto weights
        projs = np.dot(X_train, weight)
        projs = projs.reshape(projs.shape[0],1)

        # calculate forward loss
        tm = LinearRegression()
        tm.fit(projs, y_train)

        test_proj = np.dot(X_test, weight)
        test_proj = test_proj.reshape(-1,1)
        yhat = tm.predict(test_proj)

        t_loss = loss_fun(yhat, y_test)

        return t_loss

    def train_regression(self, X_train, X_test, y_train, y_test, loss_fun, thred, max_iter):
        diff = 1000
        counter = 0

        # initialize weight
        weight = np.random.normal(0, 1, X_train.shape[1])
        weight = weight / np.sqrt(np.sum(weight ** 2))

        # construct boundaries for L-BFGS-B optimizer
        bounds = []
        for i in range(len(weight)):
            bounds.append((-1, 1))

        # iterate until converge
        while (diff > thred) and (counter < max_iter):
            o_weight = weight
            k = scipy.optimize.fmin_l_bfgs_b(self.feed_forward_regression, x0=weight,
                                             args=(X_train, X_test, y_train, y_test, loss_fun),
                                             bounds=bounds, approx_grad = True)
            weight = k[0]
            diff = np.sum((weight - o_weight)**2)

        return weight

