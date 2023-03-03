#include <kalman_filter_lib/particle_filter.h>

namespace multi_posture_leg_tracker {

std::default_random_engine gen;

/**
     * @brief ParticleFilter Constructor
     */
ParticleFilter::ParticleFilter(Eigen::Vector4d &x_in, double dt, double var_pos, double var_vel, double std_obs) : num_particles(0)
{
  x_ = x_in;
  delta_t = dt;
  std_noise_pos = var_pos;
  std_noise_vel = var_vel;
  std_noise_obs = std_obs;

  std::normal_distribution<double> N_x_init(0, std_noise_pos);
  std::normal_distribution<double> N_y_init(0, std_noise_pos);
  std::normal_distribution<double> N_vel_x_init(0, std_noise_vel);
  std::normal_distribution<double> N_vel_y_init(0, std_noise_vel);
  double n_x, n_y, n_vel_x, n_vel_y;
  n_x = N_x_init(gen);
  n_y = N_y_init(gen);
  n_vel_x = N_vel_x_init(gen);
  n_vel_y = N_vel_y_init(gen);
  initialize(x_[0] + n_x, x_[1] + n_y, x_[2] + n_vel_x, x_[3] + n_vel_y);
}

/**
         * @brief Initializes particle filter by initializing particles to Gaussian
         *   distribution around first position and all the weights to 1.
         */
void ParticleFilter::initialize(double x, double y, double vel_x, double vel_y)
{
  num_particles =1000;

  weights.resize(num_particles);
  particles.resize(num_particles);

  // Normal distribution for x, y and theta
  std::normal_distribution<double> dist_x(x, std_noise_pos); // mean is centered around the new measurement
  std::normal_distribution<double> dist_y(y, std_noise_pos);
  std::normal_distribution<double> dist_vel_x(vel_x, std_noise_vel);
  std::normal_distribution<double> dist_vel_y(vel_y, std_noise_vel);

  // create particles and set their values
  for(int i=0; i<num_particles; ++i){
    Particle p;
    p.id = i;
    p.x = dist_x(gen); // take a random value from the Gaussian Normal distribution and update the attribute
    p.y = dist_y(gen);
    p.vel_x = dist_vel_x(gen);
    p.vel_y = dist_vel_y(gen);
    p.weight = 1;

    particles[i] = p;
    weights[i] = p.weight;
  }
}

/**
         * @brief  Predicts the state for the next time step
         *   using the process model.
*/
void ParticleFilter::prediction(double vel_x, double vel_y) {
  // Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  for(int i=0; i<num_particles; ++i)
  {
    Particle *p = &particles[i]; // get address of particle to update

    // use the prediction equations from the Lesson 14
    double new_x = p->x + vel_x*delta_t;
    double new_y = p->y + vel_y*delta_t;
    double new_vel_x= p->vel_x;
    double new_vel_y= p->vel_y;


    // add Gaussian Noise to each measurement
    // Normal distribution for x, y and theta
    std::normal_distribution<double> dist_x(new_x, std_noise_pos);
    std::normal_distribution<double> dist_y(new_y, std_noise_pos);
    std::normal_distribution<double> dist_vel_x(new_vel_x, std_noise_vel);
    std::normal_distribution<double> dist_vel_y(new_vel_y, std_noise_vel);

    // update the particle attributes
    p->x = dist_x(gen);
    p->y = dist_y(gen);
    p->vel_x = dist_vel_x(gen);
    p->vel_y = dist_vel_y(gen);
  }
}

/**
         * @brief  Updates the weights for each particle based on the likelihood of the
         *   observed measurements.*/
std::vector<double> ParticleFilter::updateWeights(Eigen::Vector2d &observations) {
  // Update the weights of each particle using a multi-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  std::normal_distribution<double> N_obs_x(0, std_noise_obs);
  std::normal_distribution<double> N_obs_y(0, std_noise_obs);
  double n_x, n_y;
  double weights_sum= 0;

  n_x = N_obs_x(gen);
  n_y = N_obs_y(gen);
  double obs_x = observations[0] + n_x;
  double obs_y = observations[1] + n_y;
  double probability_max =0;
  double temp_x = 0;
  double temp_y = 0;
  for(int i=0; i<num_particles; ++i)
  {
    double probability = 1;

    Particle *p = &particles[i];
    double dx = observations[0]  - p->x;
    double dy = observations[1]  - p->y;
    probability *= 1.0/(2*M_PI*std_noise_obs*std_noise_obs) * exp(-dx*dx / (2*std_noise_obs*std_noise_obs))* exp(-dy*dy / (2*std_noise_obs*std_noise_obs));
    if (probability>probability_max)
    {
      temp_x = p->x;
      temp_y = p->y;
      probability_max = probability;
    }

    weights_sum += probability;
    p->weight = probability;
    weights[i] = probability;
  }
  for(int i=0; i<num_particles; ++i)
  {
    Particle *p = &particles[i];
    p->weight /= weights_sum;
    weights[i] = p->weight;
  }
  std::vector<double> near_part;
  near_part.push_back(temp_x);
  near_part.push_back(temp_y);
  return near_part;
}

/**
         * @brief Resamples from the updated set of particles to form
         *   the new set of particles.
         */
void ParticleFilter::resample()
{
  // Resample particles with replacement with probability proportional to their weight.

  // Random integers on the [0, n) range
  // the probability of each individual integer is its weight of the divided by the sum of all weights.
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());
  std::vector<Particle> resampled_particles;

  for (int i = 0; i < num_particles; i++)
  {
    resampled_particles.push_back(particles[distribution(gen)]);
  }

  particles = resampled_particles;
}

/**
     * @brief ParticleFilter Copy Constructor
     */
ParticleFilter::ParticleFilter(const ParticleFilter &Kalman)
{
  x_=Kalman.x_;
}

}
