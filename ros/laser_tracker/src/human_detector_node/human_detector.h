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
#ifndef LASER_HUMAN_TRACKER_HUMAN_DETECTOR_H
#define LASER_HUMAN_TRACKER_HUMAN_DETECTOR_H

// ROS
#include <ros/ros.h>

#include <tf/transform_listener.h>

#include <sensor_msgs/LaserScan.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/ml/ml.hpp>

// Local headers
#include <laser_squat_leg_tracker/cluster_features.h>
#include <laser_squat_leg_tracker/laser_processor.h>

/// @todo: use pluginlib to register the specific classifier
#include <classifier_lib/classifier.h>
#include <classifier_lib/opencv_classifiers/opencv_random_forest_classifier.h>
#include <classifier_lib/opencv_classifiers/opencv_adaboost_classifier.h>
#include <classifier_lib/mlpack_classifiers/mlpack_random_forest_classifier.h>
#include <classifier_lib/mlpack_classifiers/mlpack_adaboost_classifier.h>

// Custom messages
#include <rc_tracking_msgs/Leg.h>
#include <rc_tracking_msgs/LegArray.h>

namespace multi_posture_leg_tracker
{

namespace laser_human_tracker
{
/**
 * @brief Detects clusters in laser scan and classify them as legs or squatting persons or backgrounds
 */
class HumanDetector
{
public:
  /**
   * @brief Constructor
   */
  HumanDetector(ros::NodeHandle nh);

private:
  tf::TransformListener tfl_;

  std::shared_ptr<Classifier> classifier_; /**< classifier interface that can point to any specific
                                                            classifier implementation */
  std::string classifier_type_;                         /**< type of classifier used in detection */
  int feat_count_;                                      /**< Number of features taken into account by the classifier */

  ClusterFeatures cf_;   /**< Used to calculate features of a cluster */
  int feature_set_size_; /**< size of feature set */

  int scan_num_;          /**< counter of radar scans */
  double execution_time_; /**< total run time until current frame */
  double max_exec_time_;  /**< maximum run time */
  double avg_exec_time_;  /**< average run time */

  bool use_scan_header_stamp_for_tfs_; /**< Defines which time is used*/
  bool visualize_contour_;             /**< whether the cluster contour will be visualized in Rviz */
  bool publish_background_;            /**< whether clusters classified as backgrounds will be published */

  ros::NodeHandle nh_;
  ros::Publisher markers_pub_;           /**< Publisher for the markers in rviz */
  ros::Publisher detected_clusters_pub_; /**< Publisher for detected clusters */
  ros::Subscriber scan_sub_;             /**< Laser scan subscriber */

  std::string fixed_frame_; /**< The frame, where clusters are published*/

  double detection_threshold_; /**< cluster with confidence greater than threshold can be published */
  double cluster_dist_euclid_; /**< maximal distance between 2 points in one cluster*/
  int min_points_per_cluster_; /**< minimum number of points in the cluster*/
  double max_detect_distance_; /**< Only consider clusters within max distance */
  int max_detected_clusters_;  /**< maximum number of published clusters */

  int num_prev_markers_published_; /**< counter for the markers published in last frame */

  /**
   * @brief Clusters the scan according to euclidian distance,
   *        predicts the confidence that each cluster belongs to a person and
   *        publishes the results
   *
   * Called every time a laser scan is published.
   */
  void laserCallback(const sensor_msgs::LaserScan::ConstPtr& scan);
};

/**
 * @brief Comparison class to order clusters according to their relative distance
 * to the laser scanner or their classification confidence
 */
class CompareClusters
{
public:
  bool operator()(const rc_tracking_msgs::Leg& a, const rc_tracking_msgs::Leg& b);
  bool operator()(const std::shared_ptr<rc_tracking_msgs::Leg>& a,
                  const std::shared_ptr<rc_tracking_msgs::Leg>& b);
};
}  // namespace laser_human_tracker
}
#endif  // LASER_HUMAN_TRACKER_HUMAN_DETECTOR_H
