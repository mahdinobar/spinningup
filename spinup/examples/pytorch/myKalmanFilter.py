# -*- coding: utf-8 -*-
"""
KF mnobar@ethz.ch

"""
import numpy as np


class KalmanFilter(object):

    # x0 - initial guess of the state vector 
    # P0 - initial guess of the covariance matrix of the state estimation error
    # A,B,C - system matrices describing the system model
    # Q - covariance matrix of the process noise 
    # R - covariance matrix of the measurement noise

    def __init__(self, x0, P0, A, B, C, Q, R):

        # initialize vectors and matrices
        self.x0 = x0
        self.P0 = P0
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R

        # this variable is used to track the current time step k of the estimator 
        # after every measurement arrives, this variables is incremented for +1 
        self.currentTimeStep = 0

        # this list is used to store the a posteriori estimates xk^{+} starting from the initial estimate 
        # note: list starts from x0^{+}=x0 - where x0 is an initial guess of the estimate
        # self.estimates_aposteriori=[]
        # self.estimates_aposteriori.append(x0)
        self.estimates_aposteriori = self.x0.reshape(self.x0.size, 1)

        # this list is used to store the a apriori estimates xk^{-} starting from x1^{-}
        # note: x0^{-} does not exist, that is, the list starts from x1^{-}
        self.estimates_apriori = []

        # this list is used to store the a posteriori estimation error covariance matrices Pk^{+}
        # note: list starts from P0^{+}=P0, where P0 is the initial guess of the covariance
        # self.estimationErrorCovarianceMatricesAposteriori=[]
        # self.estimationErrorCovarianceMatricesAposteriori.append(P0)
        self.estimationErrorCovarianceMatricesAposteriori = np.asarray(P0).reshape((1, self.x0.size, self.x0.size))

        # this list is used to store the a priori estimation error covariance matrices Pk^{-}
        # note: list starts from P1^{-}, that is, P0^{-} does not exist
        self.estimationErrorCovarianceMatricesApriori = []

        # this list is used to store the gain matrices Kk
        self.gainMatrices = []

        # this list is used to store prediction errors error_k=y_k-C*xk^{-}
        self.errors = []

        self.X_prediction_ahead = []

    # this function propagates x_{k-1}^{+} through the model to compute x_{k}^{-}
    # this function also propagates P_{k-1}^{+} through the covariance model to compute P_{k}^{-}
    # at the end this function increments the time index currentTimeStep for +1
    def propagateDynamics(self, inputValue):

        xk_minus = self.A * self.estimates_aposteriori[:, self.currentTimeStep].reshape(self.x0.size, 1) + self.B*inputValue.reshape(3,1)

        Pk_minus = self.A * np.asmatrix(
            self.estimationErrorCovarianceMatricesAposteriori[self.currentTimeStep, :, :]) * (self.A.T) + self.Q

        if self.currentTimeStep == 0:
            self.estimates_apriori = np.asarray(xk_minus).reshape(self.x0.size, 1)
            self.estimationErrorCovarianceMatricesApriori = np.asarray(Pk_minus).reshape((1, self.x0.size, self.x0.size))
        else:
            self.estimates_apriori = np.hstack((self.estimates_apriori, np.asarray(xk_minus).reshape(self.x0.size, 1)))
            self.estimationErrorCovarianceMatricesApriori = np.vstack(
                (self.estimationErrorCovarianceMatricesApriori, np.asarray(Pk_minus).reshape((1, self.x0.size, self.x0.size))))

        self.currentTimeStep = self.currentTimeStep + 1

    # this function should be called after propagateDynamics() because the time step should be increased and states and covariances should be propagated         
    def computeAposterioriEstimate(self, currentMeasurement):

        # if measurement_received == True:
        # gain matrix
        Kk = np.asmatrix(self.estimationErrorCovarianceMatricesApriori[self.currentTimeStep - 1, :, :]) * (
            self.C.T) * np.linalg.inv(self.R + self.C * np.asmatrix(
            self.estimationErrorCovarianceMatricesApriori[self.currentTimeStep - 1, :, :]) * (self.C.T))
        # update prediction error when measurement data received (Innovation)
        error_k = currentMeasurement.reshape(currentMeasurement.size, 1) - self.C * self.estimates_apriori[:,
                                                              self.currentTimeStep - 1].reshape(self.x0.size, 1)
        # a posteriori estimate
        xk_plus = self.estimates_apriori[:, self.currentTimeStep - 1].reshape(self.x0.size, 1) + Kk * error_k
        # a posteriori matrix update
        IminusKkC = np.matrix(np.eye(self.x0.shape[0])) - Kk * self.C
        Pk_plus = IminusKkC * self.estimationErrorCovarianceMatricesApriori[self.currentTimeStep - 1, :, :] * (
            IminusKkC.T) + Kk * (self.R) * (Kk.T)

        # else:
        #     # a posteriori estimate
        #     xk_plus = self.estimates_apriori[:, self.currentTimeStep - 1].reshape(3, 1)

        # update the lists that store the vectors and matrices
        # self.gainMatrices.append(Kk)
        if self.currentTimeStep == 1:
            self.errors = np.asarray(error_k).reshape(currentMeasurement.size, 1)
            self.gainMatrices = np.asarray(Kk).reshape((1, self.x0.size, currentMeasurement.size))
        else:
            self.errors = np.vstack((self.errors, np.asarray(error_k).reshape(currentMeasurement.size, 1)))
            self.gainMatrices = np.vstack((self.gainMatrices, np.asarray(Kk).reshape((1, self.x0.size, currentMeasurement.size))))
        self.estimates_aposteriori = np.hstack((self.estimates_aposteriori, np.asarray(xk_plus)))
        self.estimationErrorCovarianceMatricesAposteriori = np.vstack(
            (self.estimationErrorCovarianceMatricesAposteriori, np.asarray(Pk_plus).reshape(1, self.x0.size, self.x0.size)))
        # self.errors.append(error_k)
        # self.estimates_aposteriori.append(xk_plus)
        # self.estimationErrorCovarianceMatricesAposteriori.append(Pk_plus)

    def prediction_aheads(self, u, dt):
        if self.X_prediction_ahead==[]:
            self.X_prediction_ahead = self.estimates_aposteriori
        else:
            self.X_prediction_ahead =np.hstack((self.X_prediction_ahead,self.estimates_aposteriori[:, -1].reshape(self.x0.size, 1)))
        for i in range(int(dt)-1):
            x = self.A * self.X_prediction_ahead[:, -1].reshape(self.x0.size, 1) + self.B * u.reshape(3,1)
            self.X_prediction_ahead = np.hstack((self.X_prediction_ahead, x))
