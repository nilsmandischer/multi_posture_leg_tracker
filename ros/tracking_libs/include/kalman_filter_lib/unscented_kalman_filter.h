//source: https://github.com/JunshengFu/tracking-with-Unscented-Kalman-Filter/blob/master/src/ukf.cpp
#ifndef UNSCENTED_KALMAN_FILTER_H
#define UNSCENTED_KALMAN_FILTER_H
#include <eigen3/Eigen/Dense>
#include <iostream>

/**
 * @brief The UnscentedKalmanFilter class. The motion model used here is constant velocity (CV) model.
 */

namespace multi_posture_leg_tracker {

class UnscentedKalmanFilter
{
 public:

  double std_radr_; /** Radar measurement noise standard deviation radius in m*/
  double std_radrd_ ;  /** Radar measurement noise standard deviation radius change in m/s */
  int n_x_;   /**  State dimension */
  int n_aug_; /** Augmented state dimension*/
  int n_aug_sigma_;   /** Number of sigma points*/
  double lambda_;   /**  Sigma point spreading parameter*/
  double NIS_;   /**  the current NIS for radar*/
  Eigen::VectorXd weights_;   /**  Weights of sigma points*/
  Eigen::VectorXd x_;   /** State vector */
  Eigen::MatrixXd P_;   /** State covariance matrix*/
  Eigen::MatrixXd Xsig_pred_;  /**  predicted sigma points matrix*/
  Eigen::MatrixXd Xsig_aug_;   /**  augmented sigma points matrix*/
  Eigen::Matrix2d R_;   /** Measurement covariance matrix*/
  double dt; /** Time step*/
  double std_yawdd_;/** Process noise standard deviation yaw acceleration in rad/s^2*/
  double std_a_;/** Process noise standard deviation longitudinal acceleration in m/s^2*/

  /**
     * @brief UnscentedKalmanFilter Constructor
     * @param delta_t [float] : discrete time increment
     * @param var_pos [float] : postion variance
     * @param var_vel [float] : velocity variance
     * @param std_obs [float] : observation variance
     * @param para_p [float] : initialized State Covariance P_ value
     */
  UnscentedKalmanFilter(float delta_t,float var_pos,float var_vel,float std_obs,float para_p);

  /**
     * @brief UnscentedKalmanFilter CopyConstructor
     */
  UnscentedKalmanFilter(const UnscentedKalmanFilter &Kalman);

  /**
     * @brief initialize the unscented kalman filter, give the initial state
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

  /**
     * @brief Calculation of the augmented sigma points
     */
  void AugmentedSigmaPoints();
  /**
     * @brief Prediction of the sigma points
     */
  void SigmaPointPrediction();
  /**
     * @brief Prediction of the mean and covariance of the measurement
     */
  void PredictMeanAndCovariance();
};

}

#endif // UNSCENTED_KALMAN_FILTER_H
