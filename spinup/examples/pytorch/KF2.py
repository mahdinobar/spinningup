# -*- coding: utf-8 -*-
"""
Kalman filter implementation in Python and C++
Mahdi Nobar
mnobar@ethz.ch

"""
import numpy as np
import matplotlib.pyplot as plt
from myKalmanFilter import KalmanFilter



def  main_model_2(log_dir):
    xd_init = 0.5345
    yd_init = -0.2455
    zd_init = 0.1392
    x0 = np.array([xd_init,yd_init,zd_init])  # [m]

    # define the system matrices - Newtonian system
    # system matrices and covariances
    A = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    B = np.array([[0], [1], [0]])
    C = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # measurement noise covariance
    R = np.array([[2 ** 2, 0, 0], [0, 5 ** 2, 0], [0, 0, 2 ** 2]])
    # process uncertainty covariance
    Q = np.array([[1 ** 2, 0, 0], [0, 2 ** 2, 0], [0, 0, 1 ** 2]])  # np.matrix(np.zeros((3, 3)))

    # initial covariance matrix
    P0 = np.asmatrix(np.diag([1, 4, 1]))


    MAX_TIMESTEPS=136
    # Generate time stamp randomness of camera measurements
    time_randomness = np.random.normal(0, 32, 137).astype(int)
    time_randomness = np.clip(time_randomness, -49, 49)
    time_randomness[0] = np.clip(time_randomness[0], 1, 49)
    tVec_camera = np.linspace(0, 13600, 137) + time_randomness #[ms]
    vxd = np.random.normal(loc=0.0, scale=0.000367647, size=1)[
        0]  # [m/s] for 2 [cm] drift given std error after 13.6 [s]
    vyd = 34.9028e-3 + np.random.normal(loc=0.0, scale=0.002205882, size=1)[
        0]  # [m/s] for 5 [cm] drift given std error after 13.6 [s]
    vzd = 0
    x_camera = np.zeros((MAX_TIMESTEPS))
    y_camera = np.zeros((MAX_TIMESTEPS))
    z_camera = np.zeros((MAX_TIMESTEPS))

    x_camera[0] = xd_init
    y_camera[0] = yd_init
    z_camera[0] = zd_init
    dt_camera = np.hstack((tVec_camera[0], np.diff(tVec_camera)))
    for i in range(0, MAX_TIMESTEPS - 1):
        x_camera[i + 1] = x_camera[i] + vxd * dt_camera[i] + np.random.normal(loc=0.0, scale=0.0005, size=1)
        y_camera[i + 1] = y_camera[i] + vyd * dt_camera[i] + np.random.normal(loc=0.0, scale=0.001, size=1)
        z_camera[i + 1] = z_camera[i] + vzd * dt_camera[i] + np.random.normal(loc=0.0, scale=0.0005, size=1)
    X_camera = np.array([x_camera,y_camera,z_camera])


    # create a Kalman filter object
    KalmanFilterObject = KalmanFilter(x0, P0, A, B, C, Q, R)
    u = np.array([vxd,vyd,vzd])
    # simulate online prediction
    for k_measured in range(0, np.size(tVec_camera)-1):  # np.arange(np.size(tVec_camera)):
        print(k_measured)
        # TODO correct for the online application where dt is varying and be know the moment we receive the measurement
        dt = dt_camera[k_measured]
        KalmanFilterObject.B = np.array([[dt], [dt], [dt]])
        KalmanFilterObject.propagateDynamics(u)
        KalmanFilterObject.B = np.array([[1], [1], [1]])
        KalmanFilterObject.prediction_aheads(u, dt)
        KalmanFilterObject.computeAposterioriEstimate(X_camera[:, k_measured])

    # extract the state estimates in order to plot the results
    x_hat = []
    y_hat = []
    z_hat = []

    for j in range(0, np.size(tVec_camera)):
        # python estimates
        x_hat.append(KalmanFilterObject.estimates_aposteriori[0, j])
        y_hat.append(KalmanFilterObject.estimates_aposteriori[1, j])
        z_hat.append(KalmanFilterObject.estimates_aposteriori[2, j])


    tVec = np.linspace(0, (int(tVec_camera[-2]) - 1), int(tVec_camera[-2]))


    td=np.linspace(0, 13600, 137)
    xd=np.asarray(KalmanFilterObject.X_prediction_ahead[0, :]).squeeze()
    xd=xd[td[:-2].astype(int)]
    yd=np.asarray(KalmanFilterObject.X_prediction_ahead[1, :]).squeeze()
    yd=yd[td[:-2].astype(int)]
    zd=np.asarray(KalmanFilterObject.X_prediction_ahead[2, :]).squeeze()
    zd=zd[td[:-2].astype(int)]


        fig, ax = plt.subplots(3, 1, figsize=(12, 8))
        ax[0].plot(tVec_camera[0:-1], X_camera[0, :], '-or', linewidth=1, markersize=5, label='measured')
        ax[0].plot(tVec_camera[0:-1], x_hat[1:], 'ob', linewidth=1, markersize=5, label='aposteriori estimated')
        # ax[0].plot(tVec_camera, x_hat_cpp, 'om', linewidth=1, markersize=5, label='aposteriori estimated Cpp')
        # ax[0].plot(tVec, x_pred_cpp, '^y', linewidth=1, markersize=5, label='predictions ahead Cpp')
        ax[0].plot(tVec, np.asarray(KalmanFilterObject.X_prediction_ahead[0, :]).squeeze(), '-Dk', linewidth=1,
                   markersize=1, label='prediction ahead Python')
        ax[0].plot(td[:-2], xd, '-*g', linewidth=1,
                   markersize=7, label='PREDICTION FOR SIMULATION - 100 [ms]')
        ax[0].set_ylabel("x [mm]", fontsize=14)
        ax[0].legend()
        ax[1].plot(tVec_camera[0:-1], X_camera[1, :], '-or', linewidth=1, markersize=5, label='measured')
        ax[1].plot(tVec_camera[0:-1], y_hat[1:], '-ob', linewidth=1, markersize=5, label='aposteriori estimated')
        # ax[1].plot(tVec_camera, y_hat_cpp, '-om', linewidth=1, markersize=5, label='aposteriori estimated Cpp')
        # ax[1].plot(tVec, y_pred_cpp, '^y', linewidth=1, markersize=5, label='predictions ahead Cpp')
        ax[1].plot(tVec, np.asarray(KalmanFilterObject.X_prediction_ahead[1, :]).squeeze(), '-Dk', linewidth=1,
                   markersize=1, label='prediction ahead')
        ax[1].plot(td[:-2], yd, '-*g', linewidth=1,
                   markersize=7, label='PREDICTION FOR SIMULATION - 100 [ms]')
        ax[1].set_ylabel("y [mm]", fontsize=14)
        ax[1].legend()
        ax[2].plot(tVec_camera[0:-1], X_camera[2, :], '-or', linewidth=1, markersize=5, label='measured')
        ax[2].plot(tVec_camera[0:-1], z_hat[1:], '-ob', linewidth=1, markersize=5, label='aposteriori estimated')
        # ax[2].plot(tVec_camera, z_hat_cpp, '-om', linewidth=1, markersize=5, label='aposteriori estimated Cpp')
        # ax[2].plot(tVec, z_pred_cpp, '^y', linewidth=1, markersize=5, label='predictions ahead Cpp')
        ax[2].plot(tVec, np.asarray(KalmanFilterObject.X_prediction_ahead[2, :]).squeeze(), '-Dk', linewidth=1,
                   markersize=1, label='prediction ahead')
        ax[2].plot(td[:-2], zd, '-*g', linewidth=1,
                   markersize=7, label='PREDICTION FOR SIMULATION - 100 [ms]')
        ax[2].set_xlabel("$t_{k}$ [ms]", fontsize=14)
        ax[2].set_ylabel("z [mm]", fontsize=14)
        ax[2].legend()
        plt.tight_layout()
        fig.savefig('results.png', dpi=600)
        plt.show()


    np.save(log_dir + "/r_star_model_2.npy", np.asarray(KalmanFilterObject.X_prediction_ahead))
    np.savetxt(log_dir + '/r_star_model_2.csv', np.asarray(KalmanFilterObject.X_prediction_ahead), delimiter=',')
    np.savetxt(log_dir + '/x_star_model_2.csv', np.asarray(KalmanFilterObject.X_prediction_ahead)[0, :], delimiter=',')
    np.savetxt(log_dir + '/y_star_model_2.csv', np.asarray(KalmanFilterObject.X_prediction_ahead)[1, :], delimiter=',')
    np.savetxt(log_dir + '/z_star_model_2.csv', np.asarray(KalmanFilterObject.X_prediction_ahead)[2, :], delimiter=',')
    np.save(log_dir + "/t_model_2.npy", tVec)
    np.savetxt(log_dir + '/t_model_2.csv', tVec, delimiter=',')

    print("end")

if __name__ == "__main__":
    log_dir = "/home/mahdi/Downloads/draft"
    main_model_2(log_dir)

