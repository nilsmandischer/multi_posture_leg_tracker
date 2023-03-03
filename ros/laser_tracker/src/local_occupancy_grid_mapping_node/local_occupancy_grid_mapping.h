#ifndef LASER_HUMAN_TRACKER_LOCAL_OCCUPANCY_GRID_MAPPING_H
#define LASER_HUMAN_TRACKER_LOCAL_OCCUPANCY_GRID_MAPPING_H

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

// ROS messages
#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <visualization_msgs/Marker.h>

// Custom messages
#include <rc_tracking_msgs/Leg.h>
#include <rc_tracking_msgs/LegArray.h>

#include <laser_squat_leg_tracker/laser_processor.h>

#define ALPHA 0.2
#define BETA 0.1
#define OBSTACLE 0.7
#define FREE_SPACE 0.4
#define UNKNOWN 0.5
#define MIN_PROB 0.001
#define MAX_PROB 1 - MIN_PROB

namespace multi_posture_leg_tracker {

namespace laser_human_tracker
{
/**
 * @basic A simple 'local' occupancy grid map that maps everything except
 * tracked humans
 *
 * Maps a small area around the robot. The occupied areas on the map are all
 * non-human obstacles.
 */
class OccupancyGridMapping
{
public:
  OccupancyGridMapping(ros::NodeHandle nh, std::string scan_topic);

private:
  std::string scan_topic_;
  std::string fixed_frame_;
  std::string base_frame_;

  ros::NodeHandle nh_;
  message_filters::Subscriber<sensor_msgs::LaserScan> scan_sub_;
  message_filters::Subscriber<rc_tracking_msgs::LegArray> non_leg_clusters_sub_;
  message_filters::TimeSynchronizer<sensor_msgs::LaserScan, rc_tracking_msgs::LegArray> sync;
  ros::Subscriber odom_sub_;
  ros::Subscriber pose_sub_;
  ros::Publisher map_pub_;

  double l0_;
  std::vector<double> l_;
  double l_min_;
  double l_max_;

  double resolution_;
  int width_;

  bool grid_centre_pos_found_;
  double grid_centre_pos_x_;
  double grid_centre_pos_y_;
  double shift_threshold_;

  ros::Time last_time_;
  bool invalid_measurements_are_free_space_;
  double reliable_inf_range_;
  bool use_scan_header_stamp_for_tfs_;
  ros::Time latest_scan_header_stamp_with_tf_available_;
  bool unseen_is_freespace_;

  double cluster_dist_euclid_;
  int min_points_per_cluster_;

  tf::TransformListener tfl_;

  /**
   * @brief Coordinated callback for both laser scan message and a
   * non_leg_clusters message
   *
   * Called whenever both topics have been recently published to
   */
  void laserAndLegCallback(const sensor_msgs::LaserScan::ConstPtr& scan_msg,
                           const rc_tracking_msgs::LegArray::ConstPtr& non_leg_clusters);

  /**
   * @brief The logit function, i.e., the inverse of the logstic function
   */
  double logit(double p);

  /**
   * @brief The inverse of the logit function, i.e., the logsitic function
   */
  double inverseLogit(double p);

  /**
   * @brief Returns the equivilant of a passed-in angle in the -PI to PI range
   */
  double betweenPIandNegPI(double angle_in);
};
}  // namespace laser_human_tracker
}
#endif  // LASER_HUMAN_TRACKER_LOCAL_OCCUPANCY_GRID_MAPPING_H
