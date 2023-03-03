#ifndef RADAR_HUMAN_TRACKER_OTSU_FILTER_H
#define RADAR_HUMAN_TRACKER_OTSU_FILTER_H

// ROS
#include <ros/ros.h>
#include <ros/subscriber.h>
#include <ros/publisher.h>
#include <visualization_msgs/Marker.h>

// Sensor Msgs
#include <sensor_msgs/PointCloud.h>

#include <tf/transform_listener.h>
#include <geometry_msgs/PolygonStamped.h>

//#include "bound_rectangle.h"
#include <math.h>
#include <vector>
#include <fstream>

#define PI 3.14159265

namespace multi_posture_leg_tracker {
namespace otsu_filter
{
/**
 * @brief The radar filter framework
 *
 * This framework was developed in order to filter radar point clouds with special emphasis for use in robotic vision
 * and SLAM. The otsu filter filters out the points which are outside the detected room.
 */
class OtsuFilter
{
public:
  /**
   * @brief Void Constructor
   */
  OtsuFilter();

  /**
   * @brief Spatial radar filtering of the radar scan
   * @param scan [sensor_msgs::PointCloud scan] : raw radar scan
   * @param filtered_scan [sensor_msgs::PointCloud filtered_scan] : filtered radar scan
   * @param range_min [float range_min]: minimum range reading to be kept
   * @param range_max [float range_max]: maximum range reading to be kept
   * @param angle_min [float angle_min]: minimum angle reading to be kept
   * @param angle_max [float angle_max]: maximum angle reading to be kept
   */
  void filterRadar(const sensor_msgs::PointCloud& scan, sensor_msgs::PointCloud& filtered_scan, float range_min,
                   float range_max, float angle_min, float angle_max);

  /**
   * @brief Filtering radar scan by intensity
   * @param scan [sensor_msgs::PointCloud scan] : raw radar scan
   * @param filtered_scan [sensor_msgs::PointCloud filtered_scan] : filtered radar scan
   * @param intensity_min [int intensity_min]: minimum intensity of a radar scan to be kept
   */
  void filterRadarIntensity(const sensor_msgs::PointCloud& scan, sensor_msgs::PointCloud& filtered_scan,
                            int intensity_min);

  /**
   * @brief Calculates Otsu threshold of the radar scan
   * @param scan [sensor_msgs::PointCloud scan] : raw radar scan
   * @param var_identifier [int var_identifier]: defines information channel of the radar scan
   */
  int getOtsuThreshold(sensor_msgs::PointCloud* scan, int var_identifier);

private:
  // ROS system objects
  ros::NodeHandle nh;            /**< Node Handle reference from embedding node */
  ros::Subscriber pc_subscriber; /**< Raw point cloud subscriber */
  ros::Publisher pc_publisher;   /**< Filtered point cloud publisher */

  /**
   * @brief Transform a point cloud message into a point cloud
   */
  sensor_msgs::PointCloud message2Pointcloud(const sensor_msgs::PointCloudConstPtr& msg);

  /**
   * @brief Computes the Otsu gradient
   * @param histogram [std::vector<int> histogram] : histogram of the data
   * @param thereshold [int threshold]: defines possible data point where the histogram is separated
   */
  float getOtsuGradient(std::vector<int>* histogram, int threshold);

  /**
   * @brief Main function callback
   */
  void filterPointCloud(const sensor_msgs::PointCloudConstPtr& msg);

  /**
   * @brief Load parameters from parameter server
   */
  void loadParameters();

  // General parameters
  int ch_angle;                    /**< define angle channel in pointcloud */
  int ch_range;                    /**< define range channel in pointcloud */
  bool dual_mode;                  /**< activate merged scan handling */
  std::string pc_subscriber_topic; /**< define raw pointcloud subscriber topic */
  std::string pc_publisher_topic;  /**< define filtered pointcloud publisher topic */
  float min_range;                 /**< define minimum range */
  float max_range;                 /**< define maximum range */
  float min_angle;                 /**< define minimum angle */
  float max_angle;                 /**< define maximum angle */

  int ch_angle_dual; /**< angle channel in merged pointcloud */
  int ch_range_dual; /**< range channel in merged pointcloud */
};

}  // namespace otsu_filter
}
#endif  // RADAR_HUMAN_TRACKER_OTSU_FILTER_H
