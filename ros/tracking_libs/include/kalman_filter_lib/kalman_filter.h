/**
  * @file
  * @brief Tis file contains the declaration of the Kalman_Filter class.The state space
  * is [x,y,dot_x,dot_y]. The difference between this class with the other KalmanFilter used
  * in laser_tracker is the P matrix. The other difference is the update form used
  * here is Joseph form.
  * @author Xiangzhong Liu
  * @version 1.0
  * @date 04.08.2019
  */

#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <eigen3/Eigen/Dense>
#include <iostream>

/**
 * @brief The Kalman_Filter class. The motion model used here is constant velocity (CV) model.
 * This class is used for radar_tracking.
 */

namespace multi_posture_leg_tracker {

class KalmanFilter
{
public:

    /** State vector */
    Eigen::Vector4d x_;

    /** State covariance matrix*/
    Eigen::Matrix4d P_;

    /** State transition matrix (system matrix)*/
    Eigen::Matrix4d F_;

    /** Process covariance matrix*/
    Eigen::Matrix4d Q_;

    /** Measurement matrix*/
    Eigen::MatrixXd H_;

    /** Measurement covariance matrix*/
    Eigen::Matrix2d R_;

    /**
     * @brief KalmanFilter Constructor
     * @param delta_t [float] : discrete time increment
     * @param variance_position [float] : postion variance
     * @param var_vel [float] : velocity variance
     * @param observation_noise_covariance [float] : observation variance
     * @param para_p [float] : initialized State Covariance P_ value
     */
    KalmanFilter(float delta_t,float var_pos,float var_vel,float std_obs,float para_p);

    /**
    * @brief KalmanFilter Copy Constructor
    */
   KalmanFilter(const KalmanFilter &Kalman);

    /**
     * @brief Destructor
     */
    virtual ~KalmanFilter();

    /**
     * @brief initialize the kalman filter, give the initial state
     * @param x_in [Eigen::Vector4d &] : initial state
     */
    void initializeKF(Eigen::Vector4d &x_in);

    /**
     * @brief phase 1: prediction
     * @return void, update x_ and P_
     */
    void predictKF();

    /**
      * @brief phase 2: update
      * @param z [Eigen::Vector2d] : measurement vector
      */
    void updateKF(const Eigen::Vector2d &z);
};

}
#endif // KALMAN_FILTER_H
