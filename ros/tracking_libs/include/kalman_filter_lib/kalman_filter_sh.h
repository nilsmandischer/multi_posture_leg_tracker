#ifndef KALMAN_FILTER_SH_H
#define KALMAN_FILTER_SH_H

#include <eigen3/Eigen/Dense>
#include <iostream>

namespace multi_posture_leg_tracker {

class KalmanFilterSH
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
    Eigen::MatrixXd H_;

    /** Measurement covariance matrix*/
    Eigen::Matrix2d R_;

    /** @brief Kalman_Filter Constructor */
    KalmanFilterSH();


    KalmanFilterSH(float delta_t, float var_pos, float var_vel, float std_obs, float para_p);

    /** @brief initialize the kalman filter, give the initial state */
    void initializeKF(Eigen::Vector4d &x_in);

    /** @brief prediction*/
    void predictKF();

    /** @brief update */
    void updateKF(const Eigen::Vector2d &z);

    //forgetting factor in the range of 0 and 1
    float b;
    // amnestic factor
    float d_;
};

}
#endif // KALMAN_FILTER_SH_H
