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

//! @todo: Sort and categorize includes
#include <boost/math/distributions/normal.hpp>  
#include <cmath>                                
#include <random>                 

#include <tf/tf.h>  //tf
#include <tf/transform_listener.h>

// External modules
#include <kalman_filter_lib/hungarian.h>  
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
  double confidence_;
  bool is_squat_;
  bool is_matched_;

  DetectedCluster(double pos_x, double pos_y, double confidence, bool is_squat = false);

  bool operator==(const DetectedCluster& obj) const
  {
    return id_num_ == obj.id_num_;
  }
};

/**
 * @brief The ObjectTracked class: A tracked object.
 * Could be a person leg, entire person or any arbitrary object in the laser
 * scan.
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
  bool is_person_;
  ros::Time last_seen_;
  size_t times_seen_;
  size_t times_seen_consecutive_;
  size_t times_seen_as_squat_consecutive_;
  size_t times_seen_max_one_leg_consecutive_;
  cv::Matx44d filtered_state_covariances_cv_;
  double var_obs_;
  double pos_x_;
  double pos_y_;
  double vel_x_;
  double vel_y_;
  double last_pos_x_; /**< x position of last process cycle */
  double last_pos_y_; /**< y position of last process cycle */
  double init_pos_x_; /**< initial x position */
  double init_pos_y_; /**< initial y position */
  double confidence_;
  bool seen_in_current_scan_;
  int not_seen_frames_;
  float color_[3];
  size_t id_num_;
  bool deleted_;
  double dist_travelled_;  /// The distance that the track has travelled
  double single_dist_travelled_;
  double dist_to_init_pos_;
  int travelled_not_proper_;

  /**
   * @brief ObjectTracked Constructor
   * @param x
   * @param y
   * @param now
   * @param confidence
   * @param is_person
   * @param in_free_space
   * @param scan_frequency
   */
  ObjectTracked(double x, double y, ros::Time now, double confidence, bool is_person);

  ObjectTracked(const ObjectTracked& obj);

  /**
   * @brief update
   * Update our tracked object with new observations
   *
   */
  void update(double observation_x, double observation_y);

  void predict();

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
    std::string fixed_frame;
    float max_leg_pairing_dist;
    float confidence_threshold_to_maintain_track;
    bool publish_occluded;
    bool publish_candidates;
    std::string publish_people_frame;
    bool use_scan_header_stamp_for_tfs;
    float dist_travelled_together_to_initiate_leg_pair;
    float scan_frequency;
    float confidence_percentile;
    float max_std;
    bool dynamic_mode;
  } params_;

  // ROS Members
  ros::NodeHandle nh_;
  // ROS publishers and subscribers
  ros::Publisher people_tracked_pub_;
  ros::Publisher marker_pub_;
  ros::Publisher people_pose_pub_;
  ros::Subscriber detected_clusters_sub_;

  // Space
  tf::TransformListener listener_;

  // Hungarian
  HungarianAlgorithm hungarian_algorithm_;
  double mahalanobis_dist_gate_;
  double max_cov_;
  ros::Time latest_scan_header_stamp_with_tf_available;

  // Tracking
  std::vector<std::shared_ptr<ObjectTracked>> objects_tracked_;
  std::set<std::pair<std::shared_ptr<ObjectTracked>, std::shared_ptr<ObjectTracked>>> potential_leg_pairs_;
  std::map<std::pair<std::shared_ptr<ObjectTracked>, std::shared_ptr<ObjectTracked>>, std::pair<double, double>>
      potential_leg_pair_initial_dist_travelled_;  /// @todo: The travelled distance when tracks are considered as
                                                   /// potential leg pair.
  // people_tracked not needed?
  size_t prev_track_marker_id_;
  size_t prev_person_marker_id_;
  ros::Time prev_time_;

  // Time
  int msg_num_;
  double execution_time_;
  double max_exec_time_;
  double avg_exec_time_;

  /**
   * @brief Callback for every time detect_leg_clusters publishes new sets of
   * detected clusters. It will try to match the newly detected clusters with
   * tracked clusters from previous frames.
   * @param msg
   */
  void detectedClustersCallback(const rc_tracking_msgs::LegArray& detected_clusters_msg);

  bool checkDeletion(const std::shared_ptr<ObjectTracked>& track);

  bool checkInitialization(const std::shared_ptr<ObjectTracked>& track);

  /**
   * @brief Determine the degree to which the position (x,y) is in freespace accoring to our local map
   * @param x
   * @param y
   * @return degree to which the position (x,y) is in freespace (range: 0.-1.)
   */
  //  double how_much_in_free_space(double x, double y);

  /**
   * @brief Local map callback to update our local map with a newly published one
   * @param msg The occupancy map
   */
  //  void local_map_callback(const nav_msgs::OccupancyGrid& msg);

  /**
   * @brief Match detected objects to existing object tracks using a global
   * nearest neighbour data association
   * @param objects_tracked
   * @param objects_detected
   */
  std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>>
  match_detections_to_tracks_GNN(const std::vector<std::shared_ptr<ObjectTracked>>& objects_tracked,
                                 const std::vector<std::shared_ptr<DetectedCluster>>& objects_detected);

  /**
   * @brief Publish markers of tracked objects to Rviz
   */
  void publish_tracked_objects(ros::Time now);

  /**
   * @brief Publish markers of tracked people to Rviz and to <people_tracked>
   * topic
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

}}// namespace laser_human_tracker
#endif  // LASER_HUMAN_TRACKER_HUMAN_TRACKER_H
