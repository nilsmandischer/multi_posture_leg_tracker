#include <kalman_filter_lib/kalman_filter_sh.h>

namespace multi_posture_leg_tracker{

KalmanFilterSH::KalmanFilterSH(float delta_t,float var_pos,float var_vel,float std_obs,float para_p)
{

  //var_vel is not used
  int dim_state = 4;
  int dim_out = 2;
  float dt2 = delta_t*delta_t;
  float dt3 = dt2*delta_t;
  float dt4 = dt3*delta_t;
  b= 0.95;

  //initialize state transistion matrix
  F_ = Eigen::MatrixXd::Identity(dim_state,dim_state);
  F_.topRightCorner(dim_state/2,dim_state/2) = delta_t * Eigen::MatrixXd::Identity(dim_state/2,dim_state/2);

  // initialize process covariance matrix noise 
  Q_.resize(dim_state,dim_state);
  Q_.topLeftCorner(dim_state/2,dim_state/2) = Eigen::MatrixXd::Identity(dim_state/2,dim_state/2);
  Q_.topRightCorner(dim_state/2,dim_state/2) = delta_t * Eigen::MatrixXd::Identity(dim_state/2,dim_state/2);
  Q_.bottomLeftCorner(dim_state/2,dim_state/2) = Eigen::MatrixXd::Zero(dim_state/2, dim_state/2);
  Q_.bottomRightCorner(dim_state/2,dim_state/2) = Eigen::MatrixXd::Identity(dim_state/2,dim_state/2);

  //initialize measurement covariance matrix
  R_ = std_obs*std_obs*Eigen::MatrixXd::Identity(dim_out,dim_out);

  // initialize measurement matrix
  H_.resize(dim_out,dim_state);
  H_ << Eigen::MatrixXd::Identity(dim_out,dim_out),Eigen::MatrixXd::Zero(dim_out,(dim_state-dim_out));

  // initialize state covariance matrix
  P_ =  Eigen::MatrixXd::Identity(dim_state,dim_state);

  // initialize state vector
  x_ << 0,0,0,0;
}

/**
 * @brief initialize the kalman filter with the initial state
 */
void KalmanFilterSH::initializeKF(Eigen::Vector4d &x_in)
{
  x_ = x_in;
}

/**
 * @brief phase 1: prediction
 */

void KalmanFilterSH::predictKF()
{
  x_ = F_ * x_;
  Eigen::MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

/**
 * @brief phase 2: update
 */
void KalmanFilterSH::updateKF(const Eigen::Vector2d &z)
{
  Eigen::VectorXd z_pred = H_ * x_;
  Eigen::VectorXd y = z - z_pred;
  Eigen::MatrixXd Ht = H_.transpose();
  Eigen::MatrixXd PHt = P_ * Ht;
  Eigen::MatrixXd S = H_ * PHt + R_;
  Eigen::MatrixXd Si = S.inverse();
  Eigen::MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x_size, x_size);
  Eigen::MatrixXd Temp = I- K * H_;

  //standard kalman gain
  P_ = Temp * P_;

  d_ = d_/(d_+b); //d: amnestic factor //b: forgetting factor in the range of 0 and 1

  R_ = (1-d_)*R_ + d_*(y*y.transpose() - H_*P_*Ht);
}

}
