#ifndef RADAR_HUMAN_TRACKER_EXTRACT_FEATURE_MATRIX_H
#define RADAR_HUMAN_TRACKER_EXTRACT_FEATURE_MATRIX_H

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <boost/foreach.hpp>

#include <opencv2/core/core.hpp>
//#include <opencv2/cv.hpp>
#include <opencv2/ml/ml.hpp>

#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/PoseArray.h>

#include "../human_detector_node/human_detector.h"
#include "../otsu_filter_node/otsu_filter.h"
#include <vector>

namespace multi_posture_leg_tracker {
namespace radar_human_tracker
{
/**
 * @brief The ExtractFeatureMatrix class extracts feature matrix from training data samples
 * in order to speed up training procedure
 */
class ExtractFeatureMatrix
{
public:
  /**
   * @brief Constructor
   */
  ExtractFeatureMatrix(ros::NodeHandle nh);

  /**
   * @brief run after data have been loaded
   */
  void run();

  /**
   * @brief Prase command line arguments and load training data
   * @param argc Command line arguments
   * @param argv Command line arguments
   */
  void loadData(int argc, char** argv);

private:
  int feature_set_size_; /**< Size of feature set */

  double cluster_dist_euclid_;
  int min_points_per_cluster_;
  int undersample_negative_factor_;
  double range_min_;
  double range_max_;

  std::vector<std::vector<float>> data_;

  std::string save_file_path_;

  /**
   * @brief Load annotated training data from a rosbag
   * @param rosbag_file ROS bag to load data from
   * @param scan_topic Scan topic we should draw the data from in the ROS bag
   * @param data All loaded data is returned in this var
   *
   * Separate the scan into clusters, figure out which cluster lies near a
   * marker, calcualte features on those clusters, save features of
   * each cluster to <data>.
   */
  void loadAnnotatedData(const char* rosbag_file, const char* scan_topic, const char* cluster_topic,
                         std::vector<std::vector<float>>& data);

  /**
   * @brief Load negative training data from a rosbag
   * @param rosbag_file ROS bag to load data from
   * @param scan_topic Scan topic we should draw the data from in the ROS bag
   * @param data All loaded data is returned in this var
   *
   *  Load scan messages from the rosbag_file which contains only non-human objects, separate into clusters,
   * calcualte features on those clusters, save features from each cluster to <data>.
   */
  void loadNegData(const char* rosbag_file, const char* scan_topic, std::vector<std::vector<float>>& data);

  /**
   * @brief saveCvMatrix Save the extracted feature matrix in a file.
   * @param data Extracted feature matrix
   */
  void saveCvMatrix(const std::vector<std::vector<float>>& data);
};

}  // namespace radar_human_tracker
} 
#endif  // RADAR_HUMAN_TRACKER_EXTRACT_FEATURE_MATRIX_H
