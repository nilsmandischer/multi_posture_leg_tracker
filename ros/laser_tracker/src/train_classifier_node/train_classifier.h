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

#ifndef LASER_HUMAN_TRACKER_TRAIN_CLASSIFIER_H
#define LASER_HUMAN_TRACKER_TRAIN_CLASSIFIER_H

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <boost/foreach.hpp>

#include <opencv2/core/core.hpp>
//#include <opencv2/cv.hpp>
#include <opencv2/ml/ml.hpp>

#include <geometry_msgs/PoseArray.h>
#include <sensor_msgs/LaserScan.h>

#include <laser_squat_leg_tracker/cluster_features.h>
#include <laser_squat_leg_tracker/laser_processor.h>

#include <classifier_lib/classifier.h>
#include <classifier_lib/opencv_classifiers/opencv_random_forest_classifier.h>
#include <classifier_lib/opencv_classifiers/opencv_adaboost_classifier.h>
#include <classifier_lib/mlpack_classifiers/mlpack_random_forest_classifier.h>
#include <classifier_lib/mlpack_classifiers/mlpack_adaboost_classifier.h>

namespace multi_posture_leg_tracker
{

namespace laser_human_tracker
{
/**
 * @brief Trains the classifier to classify scan clusters as legs/squatting persons/backgrounds
 */
class TrainClassifier
{
public:
  /**
   * @brief Constructor
   */
  TrainClassifier(ros::NodeHandle nh);

  /**
   * @brief run after data have been loaded
   */
  void run();

  /**
   * @brief Prase command line arguments and load training and test data
   * @params argc Command line arguments
   * @params argv Command line arguments
   */
  void loadData(int argc, char** argv);

private:
  int mode_;
  std::string classifier_type_;
  std::shared_ptr<Classifier> classifier_;

  ClusterFeatures cf_;
  int feature_set_size_;

  double cluster_dist_euclid_;
  int min_points_per_cluster_;
  int undersample_negative_factor_;

  std::vector<std::vector<float>> train_leg_data_;
  std::vector<std::vector<float>> train_squat_data_;
  std::vector<std::vector<float>> train_neg_data_;
  std::vector<std::vector<float>> test_leg_data_;
  std::vector<std::vector<float>> test_squat_data_;
  std::vector<std::vector<float>> test_neg_data_;

  std::string model_file_;
  std::string save_file_path_;

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
};
}  // namespace laser_human_tracker
}
#endif  // LASER_HUMAN_TRACKER_TRAIN_CLASSIFIER_H
