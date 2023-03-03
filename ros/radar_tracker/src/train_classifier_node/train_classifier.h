/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#ifndef RADAR_HUMAN_TRACKER_TRAIN_CLASSIFIER_H
#define RADAR_HUMAN_TRACKER_TRAIN_CLASSIFIER_H

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <boost/foreach.hpp>

#include <opencv2/core/core.hpp>
//#include <opencv2/cv.hpp>
#include <opencv2/ml.hpp>

#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/PoseArray.h>

#include "../human_detector_node/human_detector.h"
#include "../otsu_filter_node/otsu_filter.h"
#include <vector>

#include <classifier_lib/classifier.h>
#include <classifier_lib/opencv_classifiers/opencv_random_forest_classifier.h>
#include <classifier_lib/opencv_classifiers/opencv_adaboost_classifier.h>
#include <classifier_lib/mlpack_classifiers/mlpack_random_forest_classifier.h>
#include <classifier_lib/mlpack_classifiers/mlpack_adaboost_classifier.h>

namespace multi_posture_leg_tracker {
namespace radar_human_tracker
{
/**
 * @brief Trains the classifier to classify scan clusters as legs/squatting persons/backgrounds
 * Data can be extracted from rosbag files
 * or directly come from .xml file (pre-extracted by extract_feature_matrix_node)
 */
class TrainClassifier
{
public:
  /**
   * @brief Constructor
   */
  TrainClassifier(ros::NodeHandle& nh);

  /**
   * @brief Prase command line arguments and load training and test data
   * @param argc Command line arguments
   * @param argv Command line arguments
   */
  void loadData(int argc, char** argv);

  /**
   * @brief run The main logic to train or test a classifier
   *
   * call this function after loadData
   */
  void run();

private:
  /** Parameters for filtering*/
  double range_min_; /**< minimum range reading to be kept*/
  double range_max_; /**< maximum range reading to be kept*/

  /** Parameters for Nearest Neighbour Clustering*/
  double cluster_dist_euclid_; /**< Maximum distance between two points in a cluster*/
  int min_points_per_cluster_; /**< define minimum points for a cluster for clustering*/

  /** Parameters defining the scanning area of the radar*/
  float opening_angle_min_; /**< Minimum recording angle ofthe radar*/
  float opening_angle_max_; /**< Maximum recording angle ofthe radar*/

  int mode_;                                             //!< 0 for training mode; 1 for testing mode
  std::string classifier_type_;                          //!< the type of classifier to be trained or tested
  std::shared_ptr<Classifier> classifier_;  //!< classifier interface that can point to any specific
                                                         //!< classifier implementation
  int feature_set_size_;                                 /**< Size of features per cluster */
  int undersample_negative_factor_;                      /**< factor to undersample the negative examples */
  std::vector<std::vector<float>> train_leg_data_;       /**< leg training samples*/
  std::vector<std::vector<float>> train_squat_data_;     /**< squat training samples*/
  std::vector<std::vector<float>> train_neg_data_;       /**< negative training samples*/
  std::vector<std::vector<float>> test_leg_data_;        /**< leg testing samples*/
  std::vector<std::vector<float>> test_squat_data_;      /**< squat testing samples*/
  std::vector<std::vector<float>> test_neg_data_;        /**< negative testing samples*/

  std::string model_file_;     //!< file path of trained model to be loaded and tested
  std::string save_file_path_; /**< file path of model to be saved after training */

  ros::NodeHandle nh_; /**< Node Handle reference from embedding node*/

  /**
   * @brief Load training data that contains annotated ground truth cluster positions (leg, squat or neg)
   * @param rosbag_file ROS bag to load data from
   * @param scan_topic Scan topic we should draw the data from in the ROS bag
   * @param cluster_topic Cluster topic contains positions of annotated ground truth clusters
   * @param data All loaded data is returned in this var
   *
   * Separate the scan into clusters, figure out which clusters lie near a marker,
   * calcualte features on those clusters, save features from each cluster to <data>.
   */
  void loadAnnotatedData(const char* rosbag_file, const char* scan_topic, const char* cluster_topic,
                         std::vector<std::vector<float>>& data);

  /**
   * @brief Load negative training data from a rosbag which contains only negative clusters
   * @param rosbag_file ROS bag to load data from
   * @param scan_topic Scan topic we should draw the data from in the ROS bag
   * @param data All loaded data is returned in this var
   *
   * Load scan messages from the rosbag_file, separate into clusters,
   * calcualte features on those clusters, save features from each cluster to <data>.
   */
  void loadNegData(const char* rosbag_file, const char* scan_topic, std::vector<std::vector<float>>& data);

  /**
   * @brief loadCvFeatureMatrix Load data directly from pre-extracted feature matrix
   * @param file .xml file output by extract_feature_matrix_node
   * @param data
   */
  void loadCvFeatureMatrix(const char* file, std::vector<std::vector<float>>& data);
};
}  // namespace radar_human_tracker
}
#endif  // RADAR_HUMAN_TRACKER_TRAIN_CLASSIFIER_H
