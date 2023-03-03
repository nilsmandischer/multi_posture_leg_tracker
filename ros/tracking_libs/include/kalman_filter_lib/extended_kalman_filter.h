#ifndef EXTENDED_KALMAN_FILTER_H
#define EXTENDED_KALMAN_FILTER_H
#include "eigen3/Eigen/Dense"
#include "ros/ros.h"
#include <iostream>

namespace multi_posture_leg_tracker {

class ExtendedKalmanFilter
{
public:

    /** State vector */
    Eigen::Vector4d x_;

    /** State covariance matrix*/
    Eigen::Matrix4d P_;

    /** State transition matrix*/
    Eigen::Matrix4d F_;

    /** Process covariance matrix*/
    Eigen::Matrix4d Q_;

    /** Measurement matrix*/
    Eigen::MatrixXd Hj_;

    /** Measurement covariance matrix*/
    Eigen::MatrixXd R_;

    /** @brief ExtendedKalmanFilter void Constructor */
    ExtendedKalmanFilter();

    /**
     * @brief ExtendedKalmanFilter Constructor
     * @param delta_t [float] : discrete time increment
     * @param variance_position [float] : postion variance
     * @param var_vel [float] : velocity variance
     * @param observation_noise_covariance [float] : observation variance
     * @param para_p [float] : initialized State Covariance P_ value
     */
    ExtendedKalmanFilter(float delta_t,float variance_position,float var_vel,float observation_noise_covariance,float para_p);

    /** @brief ExtendedKalmanFilter Copy Constructor */
    ExtendedKalmanFilter(const ExtendedKalmanFilter &Kalman);

    /** @brief initialize the extended kalman filter, give the initial state */
    void initializeKF(Eigen::Vector4d &x_in);

    /** @brief prediction*/
    void predictKF();

    /** @brief update */
    void updateKF(const Eigen::Vector2d &z);

    /** Calculate the Jacobian for non-linear function.
    * @param x_state : vector containing position and velocity
    * @return : jacobian */
    Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd& x_state);

    /** Checks to see if a value is near zero.
    * @param value : value to check
    * @param epsilon : near zero value to compare input against
    * @return : value if not between +/- epsilon else epsilon */
     double isZero(const double &value, const double epsilon = 1.0e-4);
};

}
#endif // EXTENDED_KALMAN_FILTER_H
