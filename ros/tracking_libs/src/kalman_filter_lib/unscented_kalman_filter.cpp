#include <kalman_filter_lib/unscented_kalman_filter.h>

namespace multi_posture_leg_tracker {

/**
     * @brief UnscentedKalmanFilter Constructor
     */
UnscentedKalmanFilter::UnscentedKalmanFilter(float delta_t,float variance_position,float var_vel,float observation_noise_covariance,float para_p)
{
   // Radar measurement noise standard deviation radius in m
   std_radr_ = sqrt(variance_position);

   // Radar measurement noise standard deviation radius change in m/s
   std_radrd_ = sqrt(var_vel);

   // State dimension: [pos_x pos_y vel_x vel_y]
   n_x_ = 4; 

   // Augmented state dimension
   n_aug_ = 7;

   // Number of sigma points
   n_aug_sigma_ = 2*n_aug_ +1;

   // Sigma point spreading parameter
   lambda_ = 3 - n_aug_;

   // Process noise standard deviation longitudinal acceleration in m/s^2
   std_a_ = 1.3;

   // Process noise standard deviation yaw acceleration in rad/s^2
   std_yawdd_ = 0.6;

   // Weights of sigma points
   weights_= Eigen::VectorXd(n_aug_sigma_);
   double t = lambda_ + n_aug_;
   weights_(0) = lambda_ / t;
   weights_.tail(n_aug_sigma_-1).fill(0.5/t);

   // state vector
   x_ = Eigen::VectorXd(n_x_);
   x_.fill(0);

   // state covariance matrix
   P_ = Eigen::MatrixXd(n_x_, n_x_);

   // predicted sigma points matrix
   Xsig_pred_ = Eigen::MatrixXd(n_x_, n_aug_sigma_);

   // augmented sigma points matrix
   Xsig_aug_ = Eigen::MatrixXd(n_aug_,n_aug_sigma_);

   //measurement noise covariance matrix
   Eigen::MatrixXd R_ = Eigen::MatrixXd(2,2);
   R_ << std_radr_*std_radr_, 0,
       0, std_radr_*std_radr_;
   dt = delta_t;
}

/**
 * @brief initialize the kalman filter with the initial state
 */
void UnscentedKalmanFilter::initializeKF(Eigen::Vector4d &x_in)
{
  x_ = x_in;

  // initialize state covariance matrix
  P_ = Eigen::MatrixXd::Identity(n_x_,n_x_);
}

/**
     * @brief phase 1: prediction
     */
void UnscentedKalmanFilter::predictKF()
{
  AugmentedSigmaPoints();
  SigmaPointPrediction();
  PredictMeanAndCovariance();
}

/**
      * @brief phase 2: update
      */
void UnscentedKalmanFilter::updateKF(const Eigen::Vector2d &z)
{
  // set measurement dimension, radar can measure pos_x, pos_y
  int n_z_ = 2;

  // create matrix for sigma points in measurement space
  Eigen::MatrixXd Zsig = Eigen::MatrixXd(n_z_, n_aug_sigma_);

  // predicted measurement mean: z_pred
  Eigen::VectorXd z_pred = Eigen::VectorXd(n_z_);

  // measurement covariance matrix S
  Eigen::MatrixXd S = Eigen::MatrixXd(n_z_, n_z_);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    Zsig(0,i) = Xsig_pred_(0,i);
    Zsig(1,i) = Xsig_pred_(1,i);
  }

  //predicted measurement mean: z_pred
  z_pred.fill(0.0);
  for (int i=0; i < n_aug_sigma_; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < n_aug_sigma_; i++) {
    //residual
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R_;

  //The measurement dimension
  int n_z = z_pred.rows();

  //create matrix for cross correlation Tc
  Eigen::MatrixXd Tc = Eigen::MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_aug_sigma_; i++) {  //2n+1 simga points

    //residual
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
    // state difference
    Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  Eigen::MatrixXd K = Tc * S.inverse();

  //residual
  Eigen::VectorXd z_diff = z - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  //Calculcate NIS value
  NIS_ = (z - z_pred).transpose() * S.inverse() * (z - z_pred);
}

/**
     * @brief Calculation of the augmented sigma points
     */
void UnscentedKalmanFilter::AugmentedSigmaPoints() {

  //create augmented mean vector
  Eigen::VectorXd x_aug_ = Eigen::VectorXd(n_aug_);
  x_aug_.fill(0.0);
  x_aug_.head(n_x_) = x_;

  //create augmented state covariance
  Eigen::MatrixXd P_aug_ = Eigen::MatrixXd(n_aug_, n_aug_);
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;
  P_aug_(n_aug_-2, n_aug_-2) = std_a_ * std_a_;
  P_aug_(n_aug_-1, n_aug_-1) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  Eigen::MatrixXd L = P_aug_.llt().matrixL();

  //create augmented sigma points
  Xsig_aug_.col(0)  = x_aug_;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug_.col(i+1)       = x_aug_ + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_+n_aug_) * L.col(i);
  }
}

/**
     * @brief Prediction of the sigma points
     */
void UnscentedKalmanFilter::SigmaPointPrediction() {

  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double pos_x = Xsig_aug_(0,i);
    double pos_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    //predicted state values
    double px_p, py_p;
    double v_p = v;
    double yaw_p = yaw + yawd*dt;
    double yawd_p = yawd;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = pos_x + v/yawd * ( sin (yaw_p) - sin(yaw));
      py_p = pos_y + v/yawd * ( cos(yaw) - cos(yaw_p) );
    }
    else {
      px_p = pos_x + v*dt*cos(yaw);
      py_p = pos_y + v*dt*sin(yaw);
    }

    //add noise
    double dt2 = dt * dt;
    px_p += 0.5*nu_a*dt2 * cos(yaw);
    py_p += 0.5*nu_a*dt2 * sin(yaw);
    v_p  += nu_a*dt;

    yaw_p  += 0.5*nu_yawdd*dt2;
    yawd_p += nu_yawdd*dt;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
  }
}

/**
     * @brief Prediction of the mean and covariance of the measurement
     */
void UnscentedKalmanFilter::PredictMeanAndCovariance() {

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    // state difference
    Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ += weights_(i) * x_diff * x_diff.transpose() ;
  }
}

}
