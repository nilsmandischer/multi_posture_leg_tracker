//source: https://github.com/shazraz/Extended-Kalman-Filter/blob/master/src/kalman_filter.cpp

#include <kalman_filter_lib/extended_kalman_filter.h>

namespace multi_posture_leg_tracker {

/**
     * @brief ExtendedKalmanFilter Constructor
     */
ExtendedKalmanFilter::ExtendedKalmanFilter(float delta_t,float variance_position,float var_vel,float observation_noise_covariance,float para_p)
{
  int dim_state = 4;
  float dt2 = delta_t*delta_t;
  float dt3 = dt2*delta_t;
  float dt4 = dt3*delta_t;

  //initialize state transistion matrix
  F_ = Eigen::MatrixXd::Identity(dim_state,dim_state);
  F_.topRightCorner(dim_state/2,dim_state/2) = delta_t * Eigen::MatrixXd::Identity(dim_state/2,dim_state/2);

  // initialize process covariance matrix noise
  Q_.resize(dim_state,dim_state);
  Q_.topLeftCorner(dim_state/2,dim_state/2) = 0.25 * variance_position*variance_position * dt4 * Eigen::MatrixXd::Identity(dim_state/2,dim_state/2);
  Q_.topRightCorner(dim_state/2,dim_state/2) = 0.5 * variance_position*variance_position * dt3 * Eigen::MatrixXd::Identity(dim_state/2,dim_state/2);
  Q_.bottomLeftCorner(dim_state/2,dim_state/2) = 0.5 * variance_position*variance_position * dt3 * Eigen::MatrixXd::Identity(dim_state/2,dim_state/2);
  Q_.bottomRightCorner(dim_state/2,dim_state/2) = dt2 * variance_position*variance_position * Eigen::MatrixXd::Identity(dim_state/2,dim_state/2);

  //initialize measurement covariance matrix
  R_ = Eigen::MatrixXd(3,3);
  R_ = observation_noise_covariance*observation_noise_covariance*Eigen::MatrixXd::Identity(3,3);

  // initialize measurement matrix
  Hj_ = Eigen::MatrixXd(3,4);

  // initialize state covariance matrix
  P_ =  para_p*para_p*Eigen::MatrixXd::Zero(dim_state,dim_state);

  // initialize state vector
  x_ << 0,0,0,0;
}

    /**
     * @brief ExtendedKalmanFilter Copy Constructor
*/
ExtendedKalmanFilter::ExtendedKalmanFilter(const ExtendedKalmanFilter &Kalman)
{
  F_=Kalman.F_;
  Q_=Kalman.Q_;
  R_=Kalman.R_;
  Hj_=Kalman.Hj_;
  P_=Kalman.P_;
  x_=Kalman.x_;
}

/**
 * @brief initialize the kalman filter with the initial state
 */
void ExtendedKalmanFilter::initializeKF(Eigen::Vector4d &x_in)
{
  x_ = x_in;
}

/**
 * @brief phase 1: prediction
 */
void ExtendedKalmanFilter::predictKF()
{
  x_ = F_ * x_;
  Eigen::MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

/**
 * @brief phase 2: update
 */
void ExtendedKalmanFilter::updateKF(const Eigen::Vector2d &z)
{
  float z_range = sqrt(z[0]*z[0] + z[1]*z[1]);
  float z_angle =  atan2(z[1],z[0]);
  Eigen::Vector3d z_new;
  z_new << z_range,z_angle, 0; //no information of velocity available

  float position_x = x_[0];
  float position_y = x_[1];
  float velocity_x = x_[2];
  float velocity_y = x_[3];

  float range = sqrt(position_x*position_x+position_y*position_y);
  float angle = atan2(position_y, position_x);
  float velocity_dash = (position_x*velocity_x + position_y*velocity_y)/range;

  if(range < 0.0001)
  {
    ROS_INFO_STREAM("Too small prediction value - reassignment of range to 0.0005");
    range = 0.0005;
  }
  const float PI = 3.14159265;

  while ((angle>PI)||(angle<-PI))
  {
      if(angle>PI)
      {
          angle -= 2*PI;
      }
      else
      {
          angle += 2*PI;
      }
  }

  Eigen::VectorXd z_pred = Eigen::VectorXd(3);
  z_pred << range, angle, velocity_dash;

  Hj_ = CalculateJacobian(x_);

  Eigen::VectorXd y = z_new - z_pred;
  Eigen::MatrixXd Hjt = Hj_.transpose();
  Eigen::MatrixXd PHjt = P_ * Hjt;
  Eigen::MatrixXd S = Hj_ * PHjt + R_;
  Eigen::MatrixXd Si = S.inverse();
  Eigen::MatrixXd K = PHjt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x_size, x_size);
  Eigen::MatrixXd Temp = I- K * Hj_;

  //standard kalman gain
  P_ = Temp * P_;
}

/**
 * @brief Calculate the Jacobian for non-linear function.
 */
Eigen::MatrixXd ExtendedKalmanFilter::CalculateJacobian(const Eigen::VectorXd& x_state)
  {
      float pos_x = x_[0];
      float pos_y = x_[1];
      float vel_x = x_[2];
      float vel_y = x_[3];

      double position_Square_Sum = isZero(pos_x*pos_x + pos_y*pos_y);
      double position_Square_Root = isZero(sqrt(position_Square_Sum));
      double position_Square_Cube = isZero(sqrt(position_Square_Sum*position_Square_Sum*position_Square_Sum));
      double position_velocity_difference1 = vel_x*pos_y - vel_y*pos_x;
      double position_velocity_difference2 = vel_y*pos_x - vel_x*pos_y;

      Eigen::MatrixXd Hj = Eigen::MatrixXd(3,4);
      Hj << pos_x/position_Square_Root, pos_y/position_Square_Root, 0, 0,
             -1.*pos_y/position_Square_Sum, pos_x/position_Square_Sum, 0 ,0,
             (pos_y*position_velocity_difference1)/position_Square_Cube, (pos_y*position_velocity_difference2)/position_Square_Cube,pos_x/position_Square_Root, pos_y/position_Square_Root;
      return Hj;
  }

  /**
 * @brief Checks to see if a value is near zero.
 */
  double ExtendedKalmanFilter::isZero(const double &value, const double epsilon)
  {
    if(value < epsilon)
    {
      return epsilon;
    }
    else
    {
      return value;
  }
}

}
