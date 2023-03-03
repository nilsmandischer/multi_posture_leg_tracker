#ifndef RADAR_HUMAN_TRACKER_HUMAN_TRACKER_H
#define RADAR_HUMAN_TRACKER_HUMAN_TRACKER_H

// ROS
#include <ros/ros.h>

// Custom Messages
#include <rc_tracking_msgs/Leg.h>
#include <rc_tracking_msgs/LegArray.h>
#include <rc_tracking_msgs/Person.h>
#include <rc_tracking_msgs/PersonArray.h>

// ROS Messages
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseArray.h>

// Math
#include <chrono>
#include <unordered_set>
#include <set>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <functional>
#include <boost/functional/hash.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <string>
#include <cstdlib>
#include <fstream>
#include <random>
#include <cmath>
#include <boost/math/distributions/normal.hpp>

#include <tf/tf.h>  //tf
#include <tf/transform_listener.h>
#include <opencv2/video/tracking.hpp>  // Kalman Filter from opencv
#include <eigen3/Eigen/Dense>

// Filters
#include <kalman_filter_lib/hungarian.h>              //For the linear sum assignment problem. Taken from: https://github.com/mcximing/hungarian-algorithm-cpp.git
#include <kalman_filter_lib/kalman_filter.h>          //"kalman_filter.h"
#include <kalman_filter_lib/extended_kalman_filter.h> //"extended_kalman_filter.h"
#include <kalman_filter_lib/unscented_kalman_filter.h>//"unscented_kalman_filter.h"
#include <kalman_filter_lib/kalman_filter_sh.h>       //"kalman_filter_sh.h"
#include <kalman_filter_lib/particle_filter.h>        //"particle_filter.h"

#define BIG_COST 100000  /// For assignment problem
#define VERY_BIG_COST BIG_COST + 1111
#define PI 3.14159265359

namespace multi_posture_leg_tracker {

namespace radar_human_tracker
{
/**
 * @brief Load parameters from parameter server
 */
void loadParameters();

static float para_pos_q;        /**< Parameter used to compute the position variance for Kalman filter Q matrix*/
static float para_vel_q;        /**< Parameter used to compute the velocity variance used for kalman filter Q matrix*/
static float para_p;            /**< Initial value for the diagonal of P matrix*/
static float delta_t;           /**< The discrete time increment*/
static float std_obs;           /**< The measurement noise variance*/
static float std_noise_pos;     /**< The psotion variance*/
static float std_noise_vel;     /**< The velocity variance*/
static std::string fixed_frame; /**< The frame of the radar point cloud*/
static std::string matching_algorithm;               /**< Choose Matching algorithm: MHT, GNN, NN, greedy NN*/
static std::string distance_metric;                  /**< Chosse distance metrci: mahalanobis or euclidian distance*/
static bool dynamic_mode;                            /**< true: For moving robot*/
static float max_leg_pairing_dist;                   /**< Maximal distance between tweo legs to form a leg pair*/
static float confidence_threshold_to_maintain_track; /**< Minimum confidence level to maintain a track*/
static std::string publish_people_frame;             /**< Frame in which the persons are published*/
static bool use_scan_header_stamp_for_tfs;           /**< Defines which time is used*/
static float scan_frequency;                         /**< Frequency in which the radar scan is published*/
static double var_obs;                               /**< Observation variance*/
static bool publish_occluded;                        /**< whether publish tracks without matched detections*/

/**<Defines a branch of tree for MHT */
struct History
{
  size_t id;              /**<ID of a branch */
  size_t first_id = 0;    /**<Id of the first detection matched */
  size_t second_id = 0;   /**<Id of the second detection matched */
  size_t third_id = 0;    /**<Id of the third detection matched */
  size_t fourth_id = 0;   /**<Id of the fourth detection matched */
  size_t fifth_id = 0;    /**<Id of the fifth detection matched */
  double probability = 1; /**<Probability of the branch */
  double dist = 0;        /**<Distance between the matched detections in consecutive frames*/
  bool new_track = true;  /**<Flag set for new created branches */
};

class DetectedCluster
{
private:
  static size_t new_cluster_id_num_; /**<Counts the number of clusters*/
public:
  size_t id_num_;     /**< Individual cluster number */
  double pos_x_;      /**< x position of a cluster */
  double pos_y_;      /**< y position of a cluster */
  double confidence_; /**< confidence of a cluster */
  bool is_matched_;   //!< True if the cluster has been matched to a track

  /**
   * @brief DetectedCluster Constructor
   * @param id_num_
   * @param pos_x_
   * @param pos_y_
   * @param confidence_
   */
  DetectedCluster(double pos_x, double pos_y, double confidence)
    : id_num_(new_cluster_id_num_), pos_x_(pos_x), pos_y_(pos_y), confidence_(confidence), is_matched_(false)
  {
    new_cluster_id_num_++;
  }

  bool operator==(const DetectedCluster& obj) const
  {
    return id_num_ == obj.id_num_;
  }
};

/**
 * @brief The ObjectTracked class: A tracked object.
 * Could be a person leg, entire person or any arbitrary object in the laser scan.
 *
 * People are tracked via a constant-velocity Kalman filter with a Gaussian acceleration distrubtion
 * Kalman filter params were found by hand-tuning.
 * A better method would be to use data-driven EM find the params.
 * The important part is that the observations are "weighted" higher than the motion model
 * because they're more trustworthy and the motion model kinda sucks
 */
class ObjectTracked
{
public:
  /**< Tracking method */
  // KalmanFilter* Tracker;
  // ExtendedKalmanFilter* Tracker;
  // UnscentedKalmanFilter* Tracker;
  KalmanFilterSH* Tracker;
  // ParticleFilter* Tracker;

  // Variables
  bool is_person_;                /**< Leg belongs to a person if 1 */
  ros::Time last_seen_;           /**< Frame in which the leg has been seen last */
  size_t times_seen_;             /**< In how many frames a leg has been seen */
  size_t times_seen_consecutive_; /**< In how many consecutive frame  a leg has been seen */
  double var_obs_;                /**< Observation variance */
  double pos_x_;                  /**< x position of a leg*/
  double pos_y_;                  /**< y position of a leg */
  double vel_x_;                  /**< x velocity of a leg */
  double vel_y_;                  /**< y velocity of a leg */
  double last_pos_x_;             /**< x position of last process cycle */
  double last_pos_y_;             /**< y position of last process cycle */
  double init_pos_x_;             /**< initial x position */
  double init_pos_y_;             /**< initial y position */
  double confidence_;             /**< Confidence if a cluster is a leg */
  bool seen_in_current_scan_;     /**< Has the leg been seen in the actual scan */
  int not_seen_frames_;           /**< In how many frames a leg has not been seen */
  float color_[3];                /**< Color of a leg */
  size_t id_num_;                 /**< Number of a leg */
  size_t id_det_;                 /**< Number of the detection initiating a leg */
  bool deleted_;                  /**< Flag for deletion of a leg */
  double dist_travelled_;         /**< Distance a leg has travelled */
  double actual_dist_travelled_;  /**< Distance a leg has travelled since the last frame */
  double dist_to_init_pos_;       //!< Distance to the initial position
  int travelled_not_proper_;      /**< Counts how often a leg has travelled less or more than a fixed threshold */
  static size_t new_leg_id_num_;  /**< Individual ID for each leg */

  /**
   * @brief ObjectTracked Constructor
   * @param x
   * @param y
   * @param now
   * @param confidence
   * @param is_person
   * @param scan_frequency
   */
  ObjectTracked(double x, double y, ros::Time now, double confidence, bool is_person, int det_num);
  /**
   * @brief ObjectTracked Default Constructor
   */
  ObjectTracked();
  /**
   * @brief ObjectTracked Copy Constructor
   */
  ObjectTracked(const ObjectTracked& obj);

  /**
   * @brief ObjectTracked Deconstructor
   */
  ~ObjectTracked(void);

  /**
   * @brief predict the state of track based on the motion model
   */
  void predict();

  /**
   * @brief update
   * Update our tracked object with new observations
   * @param observations
   */
  void update(Eigen::Vector2d observations);

  /**
   * @brief updateTraveledDist updates track's traveled distance
   */
  void updateTraveledDist();

  /**
   * @brief Load parameters from parameter server
   */
  void loadParameters();

  bool operator==(const ObjectTracked& obj) const
  {
    return id_num_ == obj.id_num_;
  }
};

class HumanTracker
{
public:
  /**
   * @brief Default constructor
   */
  HumanTracker();

private:
  /**
   * @brief Callback for every time detect_clusters publishes new sets of detected clusters.
   * It will try to match the newly detected clusters with tracked clusters from previous frames.
   * @param msg
   */
  void detectedClustersCallback(const rc_tracking_msgs::LegArray& detected_clusters_msg);

  /**
   * @brief Publish markers of tracked people to Rviz and to <people_tracked> topic
   */
  void publish_tracked_people(ros::Time now);

  /**
   * @brief checkDeletion The deletion logic of a track
   * @param track
   * @return true if the deletion logic is satisfied
   */
  bool checkDeletion(const std::shared_ptr<ObjectTracked>& track);

  /**
   * @brief checkInitialization The initialization logic of a person track
   * @param track
   * @return true if the initialization logic is satisfied
   */
  bool checkInitialization(const std::shared_ptr<ObjectTracked>& track);

  /**
   * @brief Match detected objects to existing object tracks using a global nearest neighbour data association
   * @param objects_tracked
   * @param objects_detected
   */
  std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>>
  match_detections_to_tracks_GNN(const std::vector<std::shared_ptr<ObjectTracked>>& objects_tracked,
                                 const std::vector<std::shared_ptr<DetectedCluster>>& objects_detected);

  /**
   * @brief Match detected objects to existing object tracks using a nearest neighbour data association
   * @param objects_tracked
   * @param objects_detected
   */
  std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>>
  match_detections_to_tracks_NN(const std::vector<std::shared_ptr<ObjectTracked>>& objects_tracked,
                                const std::vector<std::shared_ptr<DetectedCluster>>& objects_detected);

  /**
   * @brief Match detected objects to existing object tracks using a greedy nearest neighbour data association
   * @param objects_tracked
   * @param objects_detected
   */
  std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>>
  match_detections_to_tracks_greedy_NN(const std::vector<std::shared_ptr<ObjectTracked>>& objects_tracked,
                                       const std::vector<std::shared_ptr<DetectedCluster>>& objects_detected);

  /**
   * @brief Match detected objects to existing object tracks using a multi hypothesis tracker data association
   * @param objects_tracked
   * @param objects_detected
   */
  std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>>
  match_detections_to_tracks_MHT(const std::vector<std::shared_ptr<ObjectTracked>>& objects_tracked,
                                 const std::vector<std::shared_ptr<DetectedCluster>>& objects_detected);

  ros::NodeHandle nh_; /**< Node Handle reference from embedding node */

  // ROS publishers and subscribers
  ros::Publisher people_tracked_pub_;     /**< Tracked person publisher */
  ros::Publisher people_pose_pub_;        /**< publisher of tracked persons' pose */
  ros::Publisher marker_pub_;             /**< Marker publsiher for rviz */
  ros::Subscriber detected_clusters_sub_; /**< Susbcriber to the detected clusters */
  ros::Subscriber odom_;                  /**< Subscriber to the odometry data */

  // Matching algorithm
  HungarianAlgorithm hungarian_algorithm_;              /**< Hungarian algorithm constructor*/
  double euclidian_dist_gate;                           /**< euclidian distance gate*/
  double mahalanobis_dist_gate;                         /**< mahalanobis distance gate*/
  ros::Time latest_scan_header_stamp_with_tf_available; /**< time for the scan header*/
  const float kMax_cost_ = 9999999.f;                   /**< High cost for the cost matrix*/
  std::vector<History> track_history;                   /**< Vector with all MHT tree branches*/

  size_t prev_person_marker_id_;                                   /**< Individual id for each person*/
  std::vector<std::shared_ptr<DetectedCluster>> detected_clusters; /**< Vector with all detected cluster in the radar
                                                                      pointcloud*/
  std::vector<std::shared_ptr<ObjectTracked>> objects_tracked_; /**< Vector with all objects tracked (leg / non leg)*/
  tf::TransformListener listener_;                              /**< Transforms between different frames*/

  // Time
  int msg_num_;           /**< counter of received messages*/
  double execution_time_; /**< total run time until current frame */
  double max_exec_time_;  /**< maximum run time */
  double avg_exec_time_;  /**< average run time */
};

}  // namespace radar_human_tracker

}
#endif  // RADAR_HUMAN_TRACKER_HUMAN_TRACKER_H
