// https://github.com/mchchoi/ORIE6125/tree/master/Assignment/Project
//https://github.com/vatsl/ParticleFilter
//https://github.com/ksakmann/Particle-Filter/tree/master/src

#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <Eigen/Dense>

// boost headers

#include <iostream>
#include <string>
#include <random>
#include <boost/random.hpp>
#include <boost/math/distributions.hpp> // import all distributions
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

namespace multi_posture_leg_tracker {

struct Particle {

  int id;
  double x;
  double y;
  double vel_x;
  double vel_y;
  double weight;
};

class ParticleFilter
{
 public:

  double delta_t; /**< discrete time increment*/
  double std_noise_pos; /**< postion standard deviation*/
  double std_noise_vel; /**< velocity standard deviation*/
  double std_noise_obs; /**< observation standard deviation*/
  Eigen::Vector4d x_;   /** State vector */
  int num_particles; /** Number of particles to draw */
  std::vector<double> weights; /** Vector of weights of all particles */
  std::vector<Particle> particles; /** Set of current particles */

 public:

  /**
     * @brief ParticleFilter Constructor
     * @param x_in [Eigen::Vector4d] : state vector for initialization
     * @param delta_t [float] : discrete time increment
     * @param std_pos [float] : postion standard deviation
     * @param std_vel [float] : velocity standard deviation
     * @param std_obs [float] : observation standard deviation
     */
  ParticleFilter(Eigen::Vector4d &x_in, double delta_t, double std_pos, double std_vel, double std_obs);

  /**
     * @brief ParticleFilter Copy Constructor
     */
  ParticleFilter(const ParticleFilter &Kalman);

  /**
         * @brief  Initializes particle filter by initializing particles to Gaussian
         *   distribution around first position and all the weights to 1.
         * @param x Initial x position [m] (simulated estimate from GPS)
         * @param y Initial y position [m]
         * @param theta Initial orientation [rad]
         * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
         *   standard deviation of yaw [rad]]
         */
  void initialize(double x, double y, double vel_x, double vel_y);

  /**
         * @brief  Predicts the state for the next time step
         *   using the process model.
         * @param delta_t Time between time step t and t+1 in measurements [s]
         * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
         *   standard deviation of yaw [rad]]
         * @param velocity Velocity of car from t to t+1 [m/s]
         * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
         */
  void prediction(double vel_x, double vel_y);

  /**
         * @brief  Updates the weights for each particle based on the likelihood of the
         *   observed measurements.
         * @param sensor_range Range [m] of sensor
         * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
         *   standard deviation of bearing [rad]]
         * @param observations Vector of landmark observations
         * @param map Map class containing map landmarks
         */
  std::vector<double> updateWeights(Eigen::Vector2d &observations);

  /**
         * @brief Resamples from the updated set of particles to form
         *   the new set of particles.
         */
  void resample();
};

}

#endif // PARTICLE_FILTER_H
