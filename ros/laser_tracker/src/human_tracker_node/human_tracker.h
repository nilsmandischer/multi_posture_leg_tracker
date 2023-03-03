#ifndef LASER_HUMAN_TRACKER_HUMAN_TRACKER_H
#define LASER_HUMAN_TRACKER_HUMAN_TRACKER_H

// ROS
#include <ros/ros.h>

// Custom Messages
#include <rc_tracking_msgs/Leg.h>
#include <rc_tracking_msgs/LegArray.h>
#include <rc_tracking_msgs/Person.h>
#include <rc_tracking_msgs/PersonArray.h>

// ROS Messages
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/LaserScan.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseArray.h>

// Math
#include <boost/functional/hash.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/math/distributions/normal.hpp>  //scipy.stats
#include <cmath>                                //math
#include <random>                               //random

#include <tf/tf.h>  //tf
#include <tf/transform_listener.h>

// External modules
//#include "hungarian.h"
#include <kalman_filter_lib/hungarian.h>  //For the linear sum assignment problem. Taken from: https://github.com/mcximing/hungarian-algorithm-cpp.git
#include <opencv2/video/tracking.hpp>  // Kalman Filter from opencv

namespace multi_posture_leg_tracker
{

namespace laser_human_tracker
{
/**
 * @brief The DetectedCluster class: A detected scan cluster. Not yet associated
 * to an existing track.
 */
class DetectedCluster
{
private:
  static size_t new_cluster_id_num_;

public:
  size_t id_num_;
  double pos_x_;
  double pos_y_;
  double confidence_;     /**< confidence level of cluster to be human */
  double in_free_space_;  /**< degree of the cluster in free space */
  bool is_in_free_space_; /**< whether the cluster is in free space */
  bool is_squat_;         /**< whether the cluster is classified as squat */
  bool is_matched_;       /**< whether the cluster is matched with a track */

  DetectedCluster(double pos_x, double pos_y, double confidence, double in_free_space, bool is_squat = false);

  bool operator==(const DetectedCluster& obj) const
  {
    return id_num_ == obj.id_num_;
  }
};

/**
 * @brief The ObjectTracked class: A tracked object.
 * Could be a person  or any arbitrary object in the laser scan.
 *
 * People are tracked via a constant-velocity Kalman filter with a Gaussian
 * acceleration distrubtion Kalman filter params were found by hand-tuning. A
 * better method would be to use data-driven EM find the params. The important
 * part is that the observations are "weighted" higher than the motion model
 * because they're more trustworthy and the motion model kinda sucks
 */
class ObjectTracked
{
private:
  static size_t new_leg_id_num_;  // is globally increasing

  // Kalman
  cv::KalmanFilter kf;

  // Param
  double scan_frequency;

public:
  // Variables
  bool is_person_;                         /**< whether the track is a person track*/
  ros::Time last_seen_;                    /**< Time when the track was matched with a detection*/
  size_t times_seen_;                      /**< How many times the track is matched */
  size_t times_seen_consecutive_;          /**< For how many consecutive frames the track has been matched */
  size_t times_seen_as_squat_consecutive_; /**< For how many consecutive frames the track has been matched with a squat
                                              cluster */
  cv::Matx44d filtered_state_covariances_cv_;
  double var_obs_; /**< variance of measurement noise */
  double pos_x_;
  double pos_y_;
  double vel_x_;
  double vel_y_;
  double last_pos_x_; /**< x position of last process cycle */
  double last_pos_y_; /**< y position of last process cycle */
  double init_pos_x_; /**< initial x position */
  double init_pos_y_; /**< initial y position */
  double confidence_;
  bool seen_in_current_scan_; /**< whether the track is matched with a detection in current frame */
  int not_seen_frames_;       /**< For how many consecutive frames the track has not been matched */
  float color_[3];
  size_t id_num_;
  bool deleted_;
  double dist_travelled_;        /**< distance that the track has travelled */
  double single_dist_travelled_; /**< distance travelled in current frame */
  double dist_to_init_pos_;      /**< distance to initial position */
  int travelled_not_proper_;     /**< For how many consecutive frames the track has travelled improperly */
  double in_free_space_;         /**< degree of the track in free space */

  /**
   * @brief ObjectTracked Constructor
   */
  ObjectTracked(double x, double y, ros::Time now, double confidence, bool is_person, double in_free_space);

  ObjectTracked(const ObjectTracked& obj);

  /**
   * @brief update
   * Update tracks with new observations
   */
  void update(double observation_x, double observation_y);

  /**
   * @brief predict
   * Propagete tracks to current frame
   */
  void predict();

  /**
   * @brief updateTraveledDist
   * Update traveled distance of existing tracks
   */
  void updateTraveledDist();

  bool operator==(const ObjectTracked& obj) const
  {
    return id_num_ == obj.id_num_;
  }
};

/**
 * @brief The KalmanMultiTracker class
 *
 * Tracker for tracking all the people and objects
 */
class KalmanMultiTracker
{
private:
  // Support Members
  const float kMax_cost_ = 9999999.f;

  struct Params
  {
    std::string fixed_frame;    /**< frame where clusters are published*/
    float max_leg_pairing_dist; /**< max distance between a pair of legs */
    float confidence_threshold_to_maintain_track;
    bool publish_occluded;            /**< whether publish tracks without matched detection*/
    bool publish_candidates;          /**< whether publish tentative tracks*/
    std::string publish_people_frame; /**< frame where person tracks are published*/
    bool use_scan_header_stamp_for_tfs;
    float dist_travelled_together_to_initiate_leg_pair; /**< min distance that a leg pair has travelled to initialize a
                                                           person track*/
    float scan_frequency;
    float in_free_space_threshold; /**< used to check if a cluster or track in free space */
    float confidence_percentile;   /**< Confidence percentile for matching of clusters to tracks*/
    float max_std;                 /**< standard deviation to delete a track */
    bool dynamic_mode;             /**< true for moving robot*/
  } params_;

  // ROS Members
  ros::NodeHandle nh_;
  ros::Publisher people_tracked_pub_;      /**< Tracked person publisher */
  ros::Publisher marker_pub_;              /**< Marker publsiher for rviz */
  ros::Publisher people_pose_pub_;         /**< publisher of tracked persons' pose */
  ros::Publisher non_person_clusters_pub_; /**< publisher of non-human clusters used to update local map */
  ros::Subscriber detected_clusters_sub_;  /**< subscriber of detected clusters */
  ros::Subscriber local_map_sub_;          /**< subscriber of local map */

  tf::TransformListener listener_;
  nav_msgs::OccupancyGrid local_map_;
  bool new_local_map_received_; /**< whether new local map has been received */

  // Hungarian
  HungarianAlgorithm hungarian_algorithm_; /**< solve data association problem*/
  double mahalanobis_dist_gate_;
  double max_cov_; /**< delete tracks with covariance above max_cov_*/

  // Tracking
  std::vector<std::shared_ptr<ObjectTracked>> objects_tracked_; /**< track list */
  std::set<std::pair<std::shared_ptr<ObjectTracked>, std::shared_ptr<ObjectTracked>>> potential_leg_pairs_;
  std::map<std::pair<std::shared_ptr<ObjectTracked>, std::shared_ptr<ObjectTracked>>, std::pair<double, double>>
      potential_leg_pair_initial_dist_travelled_; /**< potential leg pairs with their initial travelled distance*/

  size_t prev_track_marker_id_;  /**< number of object track markers in previous frame*/
  size_t prev_person_marker_id_; /**< number of person track markers in previous frame*/

  // Time
  int msg_num_;           /**< counter of received messages*/
  double execution_time_; /**< total run time until current frame */
  double max_exec_time_;  /**< maximum run time */
  double avg_exec_time_;  /**< average run time */

  /**
   * @brief Callback for every time detector publishes new sets of
   * detected clusters. It will try to match the newly detected clusters with
   * tracked clusters from previous frames.
   */
  void detectedClustersCallback(const rc_tracking_msgs::LegArray& detected_clusters_msg);

  /**
   * @brief checkDeletion
   * Track deletion logic
   */
  bool checkDeletion(const std::shared_ptr<ObjectTracked>& track);

  /**
   * @brief checkInitialization
   * Track initialization logic
   */
  bool checkInitialization(const std::shared_ptr<ObjectTracked>& track);

  /**
   * @brief Determine the degree to which the position (x,y) is in freespace accoring to our local map
   * @return degree to which the position (x,y) is in freespace (range: 0.-1.)
   */
  double how_much_in_free_space(double x, double y);

  /**
   * @brief Local map callback to update our local map with a newly published one
   */
  void local_map_callback(const nav_msgs::OccupancyGrid& msg);

  /**
   * @brief Match detected objects to existing object tracks using a global nearest neighbour data association
   */
  std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>>
  match_detections_to_tracks_GNN(const std::vector<std::shared_ptr<ObjectTracked>>& objects_tracked,
                                 const std::vector<std::shared_ptr<DetectedCluster>>& objects_detected);

  /**
   * @brief Publish markers of tracked objects to Rviz
   */
  void publish_tracked_objects(ros::Time now);

  /**
   * @brief Publish markers of tracked people to Rviz and to <people_tracked> topic
   */
  void publish_tracked_people(ros::Time now);

  struct potential_leg_pair_iterator_hash
  {
    size_t operator()(
        std::set<std::pair<std::shared_ptr<ObjectTracked>, std::shared_ptr<ObjectTracked>>>::const_iterator it) const
    {
      return boost::hash<std::pair<std::shared_ptr<ObjectTracked>, std::shared_ptr<ObjectTracked>>>()(*it);
    }
  };

public:
  /**
   * @brief KalmanMultiTracker Constructor
   */
  KalmanMultiTracker();
};

}  // namespace laser_human_tracker
}
#endif  // LASER_HUMAN_TRACKER_HUMAN_TRACKER_H
