#include "human_tracker_without_map.h"

namespace multi_posture_leg_tracker
{

namespace laser_human_tracker
{
DetectedCluster::DetectedCluster(double pos_x, double pos_y, double confidence, bool is_squat)
  : id_num_(new_cluster_id_num_)
  , pos_x_(pos_x)
  , pos_y_(pos_y)
  , confidence_(confidence)
  , is_squat_(is_squat)
  , is_matched_(false)
{
  new_cluster_id_num_++;
}

ObjectTracked::ObjectTracked(double x, double y, ros::Time now, double confidence, bool is_person)
  : id_num_(new_leg_id_num_)
  , color_{ static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX),
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX),
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) }
  , is_person_(is_person)
  , last_seen_(now)
  , times_seen_(1)
  , times_seen_consecutive_(1)
  , times_seen_as_squat_consecutive_(0)
  , times_seen_max_one_leg_consecutive_(0)
  , pos_x_(x)
  , pos_y_(y)
  , vel_x_(0)
  , vel_y_(0)
  , last_pos_x_(x)
  , last_pos_y_(y)
  , init_pos_x_(x)
  , init_pos_y_(y)
  , confidence_(confidence)
  , seen_in_current_scan_(true)
  , not_seen_frames_(0)
  , deleted_(false)
  , dist_travelled_(0.0)
  , single_dist_travelled_(0.0)
  , dist_to_init_pos_(0.0)
  , travelled_not_proper_(0)
{
  new_leg_id_num_++;

  // Kalman
  ros::param::param<double>("scan_frequency", scan_frequency, 7.5);
  double delta_t = 1.0 / scan_frequency;

  double std_process_noise;
  if (scan_frequency > 7.49 && scan_frequency < 7.51)
  {
    std_process_noise = 0.06666; 
  }
  else if (scan_frequency > 9.99 && scan_frequency < 10.01)
  {
    std_process_noise = 0.05;
  }
  else if (scan_frequency > 14.99 && scan_frequency < 15.01)
  {
    std_process_noise = 1;
  }
  else
  {
    ROS_ERROR("Scan frequency needs to be either 7.5, 10 or 15 or the "
              "standard deviation of the process noise needs to be tuned to "
              "your scanner frequency");
    throw std::invalid_argument("received unacceptable scan_frequency");
  }

  double std_pos = std_process_noise;
  double std_vel = std_process_noise;
  double std_obs = 0.1;
  double var_pos = std::pow(std_pos, 2);
  double var_vel = std::pow(std_vel, 2);

  // The observation noise is assumed to be different when updating the Kalman
  // filter than when doing data association
  double var_obs_local = std::pow(std_obs, 2);
  var_obs_ = std::pow(std_obs + 0.4, 2);

  // Initiating the Kalman Filter
  kf.init(4, 2, 0, CV_64F);

  kf.statePre.at<double>(0) = x;
  kf.statePre.at<double>(1) = y;
  kf.statePre.at<double>(2) = 0;
  kf.statePre.at<double>(3) = 0;
  kf.statePost.at<double>(0) = x;
  kf.statePost.at<double>(1) = y;
  kf.statePost.at<double>(2) = 0;
  kf.statePost.at<double>(3) = 0;
  cv::setIdentity(kf.errorCovPost, cv::Scalar::all(0.5));
  cv::setIdentity(kf.errorCovPre, cv::Scalar::all(0.5));
  cv::setIdentity(filtered_state_covariances_cv_, cv::Scalar::all(0.5));

  // Constant velocity motion model
  kf.transitionMatrix = (cv::Mat_<double>(4, 4) << 1, 0, delta_t, 0, 0, 1, 0, delta_t, 0, 0, 1, 0, 0, 0, 0, 1);

  // Observation model. Can observe pos_x and pos_y (unless person is occluded).
  cv::setIdentity(kf.measurementMatrix);

  kf.processNoiseCov =
      (cv::Mat_<double>(4, 4) << var_pos, 0, 0, 0, 0, var_pos, 0, 0, 0, 0, var_vel, 0, 0, 0, 0, var_vel);

  cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(var_obs_local));
}

ObjectTracked::ObjectTracked(const ObjectTracked& obj)
  : kf(obj.kf)
  , scan_frequency(obj.scan_frequency)
  , is_person_(obj.is_person_)
  , last_seen_(obj.last_seen_)
  , times_seen_(obj.times_seen_)
  , times_seen_consecutive_(obj.times_seen_consecutive_)
  , times_seen_as_squat_consecutive_(obj.times_seen_as_squat_consecutive_)
  , times_seen_max_one_leg_consecutive_(obj.times_seen_max_one_leg_consecutive_)
  , filtered_state_covariances_cv_(obj.filtered_state_covariances_cv_)
  , var_obs_(obj.var_obs_)
  , pos_x_(obj.pos_x_)
  , pos_y_(obj.pos_y_)
  , vel_x_(obj.vel_x_)
  , vel_y_(obj.vel_y_)
  , last_pos_x_(obj.last_pos_x_)
  , last_pos_y_(obj.last_pos_y_)
  , init_pos_x_(obj.init_pos_x_)
  , init_pos_y_(obj.init_pos_y_)
  , confidence_(obj.confidence_)
  , seen_in_current_scan_(obj.seen_in_current_scan_)
  , not_seen_frames_(obj.not_seen_frames_)
  , color_{ obj.color_[0], obj.color_[1], obj.color_[2] }
  , id_num_(obj.id_num_)
  , deleted_(obj.deleted_)
  , dist_travelled_(obj.dist_travelled_)
  , single_dist_travelled_(obj.single_dist_travelled_)
  , dist_to_init_pos_(obj.dist_to_init_pos_)
  , travelled_not_proper_(obj.travelled_not_proper_)
{
  kf.controlMatrix = obj.kf.controlMatrix.clone();
  kf.errorCovPost = obj.kf.errorCovPost.clone();
  kf.errorCovPre = obj.kf.errorCovPre.clone();
  kf.gain = obj.kf.gain.clone();
  kf.measurementMatrix = obj.kf.measurementMatrix.clone();
  kf.measurementNoiseCov = obj.kf.measurementNoiseCov.clone();
  kf.processNoiseCov = obj.kf.processNoiseCov.clone();
  kf.statePost = obj.kf.statePost.clone();
  kf.statePre = obj.kf.statePre.clone();
  kf.temp1 = obj.kf.temp1.clone();
  kf.temp2 = obj.kf.temp2.clone();
  kf.temp3 = obj.kf.temp3.clone();
  kf.temp4 = obj.kf.temp4.clone();
  kf.temp5 = obj.kf.temp5.clone();
  kf.transitionMatrix = obj.kf.transitionMatrix.clone();
}

void ObjectTracked::update(double observation_x, double observation_y)
{
  cv::Mat state = kf.correct((cv::Mat_<double>(2, 1) << observation_x, observation_y));

  pos_x_ = state.at<double>(0);
  pos_y_ = state.at<double>(1);
  vel_x_ = state.at<double>(2);
  vel_y_ = state.at<double>(3);

  filtered_state_covariances_cv_ = kf.errorCovPost;
}

void ObjectTracked::predict()
{
  cv::Mat state = kf.predict();

  pos_x_ = state.at<double>(0);
  pos_y_ = state.at<double>(1);
  vel_x_ = state.at<double>(2);
  vel_y_ = state.at<double>(3);

  filtered_state_covariances_cv_ = kf.errorCovPost;
}

void ObjectTracked::updateTraveledDist()
{
  double delta_dist_travelled = std::sqrt(std::pow(pos_x_ - last_pos_x_, 2) + std::pow(pos_y_ - last_pos_y_, 2));
  if (delta_dist_travelled > 0.01)
  {
    dist_travelled_ += delta_dist_travelled;
  }

  single_dist_travelled_ = delta_dist_travelled;
  if (delta_dist_travelled < 0.02 || delta_dist_travelled > 0.2)
  {
    travelled_not_proper_++;
  }
  else
  {
    travelled_not_proper_ = 0;
  }

  dist_to_init_pos_ = std::sqrt(std::pow(pos_x_ - init_pos_x_, 2) + std::pow(pos_y_ - init_pos_y_, 2));

  last_pos_x_ = pos_x_;
  last_pos_y_ = pos_y_;
}

std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>>
KalmanMultiTracker::match_detections_to_tracks_GNN(
    const std::vector<std::shared_ptr<ObjectTracked>>& objects_tracked,
    const std::vector<std::shared_ptr<DetectedCluster>>& objects_detected)
{
  std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>> matched_tracks{};

  // Populate match_dist matrix of mahalanobis_dist between every detection
  // and every track
  std::vector<std::vector<double>> match_dist;  // matrix of probability of matching between all pepole and
                                                // all detections
  std::vector<std::shared_ptr<DetectedCluster>> eligable_detections;  // Only include detections in match_dist matrix
                                                                      // if they're in range of at least one track to
                                                                      // speed up munkres
  match_dist.reserve(objects_detected.size());
  eligable_detections.reserve(objects_detected.size());

  for (const auto& detect : objects_detected)
  {
    bool at_least_one_track_in_range = false;
    std::vector<double> new_row;
    new_row.reserve(objects_tracked.size());
    for (const auto& track : objects_tracked)
    {
      // Ignore possible matchings between people and detections not in freespace
      double cost;
      // Use mahalanobis dist to do matching
      double cov = track->filtered_state_covariances_cv_(0, 0) + track->var_obs_;  // cov_xx == cov_yy == cov
      double mahalanobis_dist =
          std::sqrt((std::pow(detect->pos_x_ - track->pos_x_, 2) + std::pow(detect->pos_y_ - track->pos_y_, 2)) /
                    cov);  
      if (mahalanobis_dist < mahalanobis_dist_gate_)
      {
        cost = mahalanobis_dist;
        at_least_one_track_in_range = true;
      }
      else
      {
        cost = kMax_cost_;
      }

      new_row.emplace_back(cost);
    }
    // If the detection is within range of at least one track, add it as an
    // eligable detection in the munkres matching
    if (at_least_one_track_in_range)
    {
      match_dist.push_back(new_row);
      eligable_detections.emplace_back(detect);
    }
  }

  // Run munkres/hungarian on match_dist to get the lowest cost assignment
  if (!match_dist.empty())
  {
    std::vector<int> assignment;
    hungarian_algorithm_.Solve(match_dist, assignment);

    for (size_t elig_detect_idx = 0; elig_detect_idx < assignment.size(); elig_detect_idx++)
    {
      if (assignment.at(elig_detect_idx) >= 0)
      {
        if (match_dist.at(elig_detect_idx).at(assignment.at(elig_detect_idx)) < mahalanobis_dist_gate_)
        {
          eligable_detections.at(elig_detect_idx)->is_matched_ = true;
          matched_tracks.emplace(objects_tracked.at(assignment.at(elig_detect_idx)),
                                 eligable_detections.at(elig_detect_idx));
        }
      }
    }
  }

  return matched_tracks;
}

void KalmanMultiTracker::publish_tracked_objects(ros::Time now)
{
  // Make sure we can get the required transform first:
  bool transform_available;
  ros::Time tf_time;
  if (params_.use_scan_header_stamp_for_tfs)
  {
    tf_time = now;
    try
    {
      listener_.waitForTransform(params_.publish_people_frame, params_.fixed_frame, tf_time, ros::Duration(1.0));
      transform_available = true;
    }
    catch (const std::exception& e)
    {
      transform_available = false;
    }
  }
  else
  {
    tf_time = ros::Time(0);
    transform_available = listener_.canTransform(params_.publish_people_frame, params_.fixed_frame, tf_time);
  }

  int marker_id = 0;
  if (!transform_available)
  {
    ROS_WARN("Laser Tracker: tf not available. Not publishing tracked objects");
  }
  else
  {
    visualization_msgs::MarkerArray markers;
    for (const auto& track : objects_tracked_)
    {
      if (track->is_person_)
      {
        continue;
      }

      if (params_.publish_occluded || track->seen_in_current_scan_)
      // Only publish people who have been seen in current scan, unless we want to publish occluded people
      {
        // Get the track position in the <params_.publish_people_frame> frame
        geometry_msgs::PointStamped ps;
        ps.header.frame_id = params_.fixed_frame;
        ps.header.stamp = tf_time;
        ps.point.x = track->pos_x_;
        ps.point.y = track->pos_y_;
        try
        {
          listener_.transformPoint(params_.publish_people_frame, ps, ps);
        }
        catch (...)
        {
          ROS_WARN_STREAM("Not publishing tracked objects due to no transform from fixed_frame: "
                          << params_.fixed_frame << "to publish_people_frame: " << params_.publish_people_frame);
          continue;
        }

        // publish rviz markers
        visualization_msgs::Marker marker;
        marker.header.frame_id = params_.publish_people_frame;
        marker.header.stamp = now;
        marker.ns = "laser_objects_tracked";
        marker.color.r = track->color_[0];
        marker.color.g = track->color_[1];
        marker.color.b = track->color_[2];
        marker.color.a = 1;
        marker.pose.position.x = ps.point.x;
        marker.pose.position.y = ps.point.y;
        marker.id = marker_id;
        marker_id++;
        marker.type = visualization_msgs::Marker::CYLINDER;
        marker.scale.x = 0.05;
        marker.scale.y = 0.05;
        marker.scale.z = 0.2;
        marker.pose.position.z = 0.15;
        markers.markers.push_back(marker);
      }
    }

    // Clear previously published track markers
    for (size_t m_id = marker_id; m_id < prev_track_marker_id_; m_id++)
    {
      visualization_msgs::Marker marker;
      marker.header.stamp = now;
      marker.header.frame_id = params_.publish_people_frame;
      marker.ns = "laser_objects_tracked";
      marker.id = m_id;
      marker.action = marker.DELETE;
      //      this->marker_pub_.publish(marker);
      markers.markers.push_back(marker);
    }
    marker_pub_.publish(markers);
    prev_track_marker_id_ = marker_id;
  }
}

void KalmanMultiTracker::publish_tracked_people(ros::Time now)
{
  rc_tracking_msgs::PersonArray people_tracked_msg;
  people_tracked_msg.header.stamp = now;
  people_tracked_msg.header.frame_id = params_.publish_people_frame;

  geometry_msgs::PoseArray people_pose_array;
  people_pose_array.header = people_tracked_msg.header;

  // Make sure we can get the required transform first:
  bool transform_available;
  ros::Time tf_time;
  if (params_.use_scan_header_stamp_for_tfs)
  {
    tf_time = now;
    try
    {
      listener_.waitForTransform(params_.publish_people_frame, params_.fixed_frame, tf_time, ros::Duration(1.0));
      transform_available = true;
    }
    catch (const std::exception& e)
    {
      transform_available = false;
    }
  }
  else
  {
    tf_time = ros::Time(0);
    transform_available = listener_.canTransform(params_.publish_people_frame, params_.fixed_frame, tf_time);
  }

  int marker_id = 0;
  visualization_msgs::MarkerArray markers;
  if (!transform_available)
  {
    ROS_WARN("Laser Tracker: tf not available. Not publishing tracked people");
  }
  else
  {
    for (auto person : objects_tracked_)
    {
      if (person->is_person_)
      {
        if (params_.publish_occluded || person->seen_in_current_scan_)
        {
          // Only publish people who have been seen in current scan, unless we want to publish occluded people
          // Get position in the <params_.publish_people_frame> frame
          geometry_msgs::PointStamped ps;
          ps.header.frame_id = params_.fixed_frame;
          ps.header.stamp = tf_time;
          ps.point.x = person->pos_x_;
          ps.point.y = person->pos_y_;
          try
          {
            listener_.transformPoint(params_.publish_people_frame, ps, ps);
          }
          catch (...)
          {
            ROS_WARN_STREAM("Not publishing tracked people due to no transform from fixed_frame: "
                            << params_.fixed_frame << "to publish_people_frame: " << params_.publish_people_frame);
            continue;
          }

          // publish to people_tracked topic
          rc_tracking_msgs::Person new_person;
          new_person.pose.position.x = ps.point.x;
          new_person.pose.position.y = ps.point.y;
          double yaw = std::atan2(person->vel_y_, person->vel_x_);
          tf::Quaternion quaternion = tf::createQuaternionFromYaw(yaw);
          new_person.pose.orientation.x = quaternion[0];
          new_person.pose.orientation.y = quaternion[1];
          new_person.pose.orientation.z = quaternion[2];
          new_person.pose.orientation.w = quaternion[3];
          new_person.id = person->id_num_;
          people_tracked_msg.people.push_back(new_person);

          // publish to people_tracked_pose_array topic
          people_pose_array.poses.push_back(new_person.pose);

          // publish rviz markers
          // Cylinder for body
          visualization_msgs::Marker marker;
          marker.header.frame_id = params_.publish_people_frame;
          marker.header.stamp = now;
          marker.ns = "laser_tracked_people";
          marker.color.r = person->color_[0];
          marker.color.g = person->color_[1];
          marker.color.b = person->color_[2];
          marker.color.a =
              (ros::Duration(3) - (ros::Time::now() - person->last_seen_)).toSec() / ros::Duration(3).toSec() + 0.1;
          marker.pose.position.x = ps.point.x;
          marker.pose.position.y = ps.point.y;
          marker.id = marker_id;
          marker_id++;
          marker.type = visualization_msgs::Marker::CYLINDER;
          marker.scale.x = 0.2;
          marker.scale.y = 0.2;
          marker.scale.z = 1.2;
          marker.pose.position.z = 0.8;
          markers.markers.push_back(marker);

          // Sphere for head shape
          marker.type = visualization_msgs::Marker::SPHERE;
          marker.scale.x = 0.2;
          marker.scale.y = 0.2;
          marker.scale.z = 0.2;
          marker.pose.position.z = 1.5;
          marker.id = marker_id;
          marker_id++;
          markers.markers.push_back(marker);

          // Text showing person's ID number
          marker.color.r = 1.0;
          marker.color.g = 1.0;
          marker.color.b = 1.0;
          marker.color.a = 1.0;
          marker.id = marker_id;
          marker_id++;
          marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
          marker.text = std::to_string(person->id_num_);
          marker.scale.z = 0.4;
          marker.pose.position.z = 1.7;
          markers.markers.push_back(marker);

          // Arrow pointing in direction they're facing with magnitude
          // proportional to speed
          marker.color.r = person->color_[0];
          marker.color.g = person->color_[1];
          marker.color.b = person->color_[2];
          marker.color.a =
              (ros::Duration(3) - (ros::Time::now() - person->last_seen_)).toSec() / ros::Duration(3).toSec() + 0.1;
          geometry_msgs::Point start_point;
          geometry_msgs::Point end_point;
          start_point.x = marker.pose.position.x;
          start_point.y = marker.pose.position.y;
          end_point.x = start_point.x + 0.5 * person->vel_x_;
          end_point.y = start_point.y + 0.5 * person->vel_y_;
          marker.pose.position.x = 0.;
          marker.pose.position.y = 0.;
          marker.pose.position.z = 0.1;
          marker.id = marker_id;
          marker_id++;
          marker.type = visualization_msgs::Marker::ARROW;
          marker.points.push_back(start_point);
          marker.points.push_back(end_point);
          marker.scale.x = 0.05;
          marker.scale.y = 0.1;
          marker.scale.z = 0.2;
          markers.markers.push_back(marker);

          // <self.confidence_percentile>% confidence bounds of person's position as an ellipse:
          double cov = person->filtered_state_covariances_cv_(0, 0) + person->var_obs_;  // cov_xx == cov_yy ==cov_zz
          double std = std::sqrt(cov);
          boost::math::normal norm_dist(0.0, std);
          double gate_dist_euclid = quantile(norm_dist, (1.0 - (1.0 - params_.confidence_percentile) / 2.0));
          marker.pose.position.x = ps.point.x;
          marker.pose.position.y = ps.point.y;
          marker.type = visualization_msgs::Marker::SPHERE;
          marker.scale.x = 2 * gate_dist_euclid;
          marker.scale.y = 2 * gate_dist_euclid;
          marker.scale.z = 0.01;
          marker.color.r = person->color_[0];
          marker.color.g = person->color_[1];
          marker.color.b = person->color_[2];
          marker.color.a = 0.1;
          marker.pose.position.z = 0.0;
          marker.id = marker_id;
          marker_id++;
          markers.markers.push_back(marker);
        }
      }
    }
  }

  for (size_t m_id = marker_id; m_id < prev_person_marker_id_; m_id++)
  {
    visualization_msgs::Marker marker;
    marker.header.stamp = now;
    marker.header.frame_id = params_.publish_people_frame;
    marker.ns = "laser_tracked_people";
    marker.id = m_id;
    marker.action = marker.DELETE;
    markers.markers.push_back(marker);
  }
  marker_pub_.publish(markers);
  prev_person_marker_id_ = marker_id;

  // Publish people tracked message
  this->people_tracked_pub_.publish(people_tracked_msg);
  people_pose_pub_.publish(people_pose_array);
}

KalmanMultiTracker::KalmanMultiTracker()
  : nh_("~")
  , prev_track_marker_id_(0)
  , prev_person_marker_id_(0)
  , msg_num_(0)
  , execution_time_(0.0)
  , max_exec_time_(0.0)
  , avg_exec_time_(0.0)
{
  // Get ROS params
  nh_.param<std::string>("fixed_frame", params_.fixed_frame, "odom");
  nh_.param<float>("max_leg_pairing_dist", params_.max_leg_pairing_dist, 0.8);
  nh_.param<float>("confidence_threshold_to_maintain_track", params_.confidence_threshold_to_maintain_track, 0.1);
  nh_.param<bool>("publish_occluded", params_.publish_occluded, true);
  nh_.param<bool>("publish_candidates", params_.publish_candidates, false);
  nh_.param<std::string>("publish_people_frame", params_.publish_people_frame, params_.fixed_frame);
  nh_.param<bool>("use_scan_header_stamp_for_tfs", params_.use_scan_header_stamp_for_tfs, false);
  nh_.param<float>("dist_travelled_together_to_initiate_leg_pair", params_.dist_travelled_together_to_initiate_leg_pair,
                   0.5);
  nh_.param<float>("scan_frequency", params_.scan_frequency, 7.5);
  nh_.param<float>("confidence_percentile", params_.confidence_percentile, 0.90);
  nh_.param<float>("max_std", params_.max_std, 0.9);
  nh_.param<bool>("dynamic_mode", params_.dynamic_mode, false);
  std::string detected_clusters_topic;
  std::string tracked_people_topic;
  std::string tracked_people_marker_topic;
  nh_.param<std::string>("detected_clusters_topic", detected_clusters_topic, "laser_detected_clusters");
  nh_.param<std::string>("tracked_people_topic", tracked_people_topic, "laser_tracked_people");
  nh_.param<std::string>("tracked_people_marker_topic", tracked_people_marker_topic, "laser_tracked_people_marker");

  boost::math::normal norm_dist(0.0, 1.0);
  mahalanobis_dist_gate_ = quantile(norm_dist, (1.0 - (1.0 - params_.confidence_percentile) / 2.0));
  max_cov_ = std::pow(params_.max_std, 2);
  latest_scan_header_stamp_with_tf_available = ros::Time::now();

  // ROS publishers
  people_tracked_pub_ = nh_.advertise<rc_tracking_msgs::PersonArray>(tracked_people_topic, 5);
  people_pose_pub_ = nh_.advertise<geometry_msgs::PoseArray>(tracked_people_topic + "_pose_array", 5);
  marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(tracked_people_marker_topic, 5);

  // ROS subscribers
  detected_clusters_sub_ =
      nh_.subscribe(detected_clusters_topic, 1, &KalmanMultiTracker::detectedClustersCallback, this);
}

void KalmanMultiTracker::detectedClustersCallback(const rc_tracking_msgs::LegArray& detected_clusters_msg)
{
  ros::Time now = detected_clusters_msg.header.stamp;

  std::vector<std::shared_ptr<DetectedCluster>> detected_clusters;
  for (auto cluster : detected_clusters_msg.legs)
  {
    detected_clusters.emplace_back(
        std::make_shared<DetectedCluster>(cluster.position.x, cluster.position.y, cluster.confidence,
                                          cluster.label == rc_tracking_msgs::Leg::LABEL_SQUAT));
  }

  std::vector<std::shared_ptr<ObjectTracked>> all_tracks;
  std::vector<std::shared_ptr<ObjectTracked>> person_tracks;
  std::vector<std::shared_ptr<ObjectTracked>> candidate_tracks;
  for (auto& track : objects_tracked_)
  {
    track->predict();
    all_tracks.emplace_back(track);
    if (track->is_person_)
    {
      person_tracks.emplace_back(track);
    }
    else
    {
      candidate_tracks.emplace_back(track);
    }
  }

  // Duplicate tracks of people so they can be matched twice in the matching
  std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<ObjectTracked>> duplicates;
  for (const auto& person_track : person_tracks)
  {
    all_tracks.emplace_back(std::make_shared<ObjectTracked>(*person_track));
    duplicates.emplace(person_track, all_tracks.back());
  }

  // Match all tracks with observations
  std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>> person_track_detection_map =
      match_detections_to_tracks_GNN(all_tracks, detected_clusters);

  std::vector<std::shared_ptr<DetectedCluster>> unmatched_detections;
  for (const auto& cluster : detected_clusters)
  {
    if (!cluster->is_matched_)
    {
      unmatched_detections.emplace_back(cluster);
    }
  }

  for (std::vector<std::shared_ptr<ObjectTracked>>::iterator it_track = objects_tracked_.begin();
       it_track != objects_tracked_.end(); it_track++)
  {
    std::shared_ptr<DetectedCluster> matched_detection(nullptr);
    if ((*it_track)->is_person_)
    {
      auto search_1 = person_track_detection_map.find(*it_track);
      auto search_2 = person_track_detection_map.find(duplicates.find(*it_track)->second);
      if (search_1 != person_track_detection_map.end() && search_2 != person_track_detection_map.end())
      {
        std::shared_ptr<DetectedCluster> md_1 = search_1->second;
        std::shared_ptr<DetectedCluster> md_2 = search_2->second;

        if (md_1->is_squat_ && md_2->is_squat_)
        {
          matched_detection = std::make_shared<DetectedCluster>(
              DetectedCluster((md_1->pos_x_ + md_2->pos_x_) / 2, (md_1->pos_y_ + md_2->pos_y_) / 2,
                              (md_1->confidence_ + md_2->confidence_) / 2, true));
        }
        else if (md_1->is_squat_)
        {
          matched_detection = md_1;
        }
        else if (md_2->is_squat_)
        {
          matched_detection = md_2;
        }
        else  // neither is squat
        {
          matched_detection = std::make_shared<DetectedCluster>(
              DetectedCluster((md_1->pos_x_ + md_2->pos_x_) / 2, (md_1->pos_y_ + md_2->pos_y_) / 2,
                              (md_1->confidence_ + md_2->confidence_) / 2, false));
        }
      }
      else if (search_1 != person_track_detection_map.end())
      {
        std::shared_ptr<DetectedCluster> md_1 = search_1->second;
        std::shared_ptr<ObjectTracked> md_2 = duplicates.find(*it_track)->second;
        if (md_1->is_squat_)
        {
          matched_detection = md_1;
        }
        else
        {
          matched_detection = std::make_shared<DetectedCluster>(DetectedCluster(
              (md_1->pos_x_ + md_2->pos_x_) / 2, (md_1->pos_y_ + md_2->pos_y_) / 2, md_1->confidence_, false));
        }
      }
      else if (search_2 != person_track_detection_map.end())
      {
        std::shared_ptr<DetectedCluster> md_1 = search_2->second;
        std::shared_ptr<ObjectTracked> md_2 = *it_track;
        if (md_1->is_squat_)
        {
          matched_detection = md_1;
        }
        else
        {
          matched_detection = std::make_shared<DetectedCluster>(DetectedCluster(
              (md_1->pos_x_ + md_2->pos_x_) / 2, (md_1->pos_y_ + md_2->pos_y_) / 2, md_1->confidence_, false));
        }
      }
      else
      {
        matched_detection = nullptr;
      }

      if (matched_detection)
      {
        (*it_track)->update(matched_detection->pos_x_, matched_detection->pos_y_);

        (*it_track)->confidence_ =
            0.8 * (*it_track)->confidence_ + 0.2 * matched_detection->confidence_; 
        (*it_track)->times_seen_++;                                                 
        (*it_track)->times_seen_consecutive_++;
        if (matched_detection->is_squat_)
        {
          (*it_track)->times_seen_as_squat_consecutive_++;
        }
        else
        {
          (*it_track)->times_seen_as_squat_consecutive_ = 0;
        }
        (*it_track)->last_seen_ = now;
        (*it_track)->seen_in_current_scan_ = true;
        (*it_track)->not_seen_frames_ = 0;
      }
      else  // no matched detection
      {
        (*it_track)->seen_in_current_scan_ = false;
        (*it_track)->times_seen_consecutive_ = 0;
        (*it_track)->times_seen_as_squat_consecutive_ = 0;
        (*it_track)->not_seen_frames_++;
        (*it_track)->times_seen_max_one_leg_consecutive_++;
      }
      (*it_track)->updateTraveledDist();

      // check Deletion logic
      if (checkDeletion(*it_track))
      {
        (*it_track)->deleted_ = true;
        objects_tracked_.erase(it_track);
        it_track--;
      }
    }

    else  // non-person track
    {
      auto search = person_track_detection_map.find(*it_track);
      if (search != person_track_detection_map.end())
      {
        matched_detection = search->second;
        (*it_track)->update(matched_detection->pos_x_, matched_detection->pos_y_);

        (*it_track)->confidence_ =
            0.8 * (*it_track)->confidence_ + 0.2 * matched_detection->confidence_;  // 0.95 & 0.05
        (*it_track)->times_seen_++;
        (*it_track)->times_seen_consecutive_++;
        if (matched_detection->is_squat_)
        {
          (*it_track)->times_seen_as_squat_consecutive_++;
        }
        else
        {
          (*it_track)->times_seen_as_squat_consecutive_ = 0;
        }
        (*it_track)->last_seen_ = now;
        (*it_track)->seen_in_current_scan_ = true;
        (*it_track)->not_seen_frames_ = 0;
      }
      else  // no matched detection
      {
        (*it_track)->seen_in_current_scan_ = false;
        (*it_track)->times_seen_consecutive_ = 0;
        (*it_track)->times_seen_as_squat_consecutive_ = 0;
        (*it_track)->not_seen_frames_++;
        (*it_track)->times_seen_max_one_leg_consecutive_++;
      }
      (*it_track)->updateTraveledDist();

      // check Deletion and Initialization logic
      if (checkDeletion(*it_track))
      {
        (*it_track)->deleted_ = true;
        objects_tracked_.erase(it_track);
        it_track--;
      }
      else if (checkInitialization(*it_track))
      {
        // checks if there has already been a person track near the candidate track.
        // in case a real person may be detected as several clusters.
        bool has_person_nearby = false;
        for (const auto& person_track : person_tracks)
        {
          if (!person_track->deleted_)
          {
            double dist = std::sqrt(std::pow((*it_track)->pos_x_ - person_track->pos_x_, 2) +
                                    std::pow((*it_track)->pos_y_ - person_track->pos_y_, 2));
            if (dist < 0.5)
            {
              has_person_nearby = true;
              break;
            }
          }
        }
        if (!has_person_nearby)
        {
          (*it_track)->is_person_ = true;
          ROS_DEBUG_STREAM("Initialize the candidate as a SQUAT person track. ID: " << (*it_track)->id_num_);
        }
      }
    }
  }

  for (const auto& detect : unmatched_detections)
  {
    auto new_track =
        std::make_shared<ObjectTracked>(ObjectTracked(detect->pos_x_, detect->pos_y_, now, detect->confidence_, false));
    if (detect->is_squat_)
    {
      new_track->times_seen_as_squat_consecutive_++;
    }
    objects_tracked_.emplace_back(new_track);
  }

  for (auto it_track_1 = objects_tracked_.begin(); it_track_1 != objects_tracked_.end(); ++it_track_1)
  {
    for (auto it_track_2 = objects_tracked_.begin(); it_track_2 != it_track_1; ++it_track_2)
    {
      if (!it_track_1->get()->is_person_ || !it_track_2->get()->is_person_)
      {
        auto temp_pair = std::make_pair(*it_track_1, *it_track_2);
        if (potential_leg_pairs_.emplace(temp_pair).second)  // if temp_pair doesn't exist in the set
        {
          auto entry_in_map = potential_leg_pair_initial_dist_travelled_.find(temp_pair);
          if (entry_in_map != potential_leg_pair_initial_dist_travelled_.end())  // if temp_pair existed already in
                                                                                 // the map.
          {
            entry_in_map->second =
                std::make_pair(it_track_1->get()->dist_travelled_, it_track_2->get()->dist_travelled_);
          }
          else  // if temp_pair is not in the map.
          {
            potential_leg_pair_initial_dist_travelled_.emplace(
                temp_pair, std::make_pair(it_track_1->get()->dist_travelled_, it_track_2->get()->dist_travelled_));
          }
        }
      }
    }
  }

  // We want to iterate over the potential leg pairs but iterating over the
  // set <self.potential_leg_pairs> will produce arbitrary iteration orders.
  // This is bad if we want repeatable tests (but otherwise, it shouldn't
  // affect performance). So we'll create a sorted list and iterate over that.
  // potential_leg_pairs is already sorted cause it's a set in c++!

  // Check if current leg pairs are still valid and if they should spawn a
  // person
  std::unordered_set<std::set<std::pair<std::shared_ptr<ObjectTracked>, std::shared_ptr<ObjectTracked>>>::iterator,
                     potential_leg_pair_iterator_hash>
      leg_pairs_to_delete;
  for (auto it = potential_leg_pairs_.begin(); it != potential_leg_pairs_.end(); ++it)
  {
    auto& track_pair = *it;
    // Check if we should delete this pair because
    //- the legs are too far apart
    //- or one of the legs has already been paired
    //- or a leg has been deleted because it hasn't been seen for a while
    double dist = std::sqrt(std::pow(track_pair.first->pos_x_ - track_pair.second->pos_x_, 2) +
                            std::pow(track_pair.first->pos_y_ - track_pair.second->pos_y_, 2));
    if (dist > params_.max_leg_pairing_dist || (track_pair.first->deleted_ || track_pair.second->deleted_) ||
        (track_pair.first->is_person_ && track_pair.second->is_person_) ||
        track_pair.first->confidence_ < params_.confidence_threshold_to_maintain_track ||
        track_pair.second->confidence_ < params_.confidence_threshold_to_maintain_track)
    {
      leg_pairs_to_delete.emplace(it);
      continue;
    }

    // Check if we should create a tracked person from this pair
    // Three conditions must be met:
    //- both tracks have been matched to a cluster in the current scan
    //- both tracks have travelled at least a distance of
    //<self.dist_travelled_together_to_initiate_leg_pair> since they were
    // paired
    //- both tracks are in free-space
    if (track_pair.first->seen_in_current_scan_ && track_pair.second->seen_in_current_scan_)
    {
      std::pair<double, double> track_pair_initial_dist(
          potential_leg_pair_initial_dist_travelled_.find(track_pair)->second);
      double dist_travelled = std::min(track_pair.first->dist_travelled_ - track_pair_initial_dist.first,
                                       track_pair.second->dist_travelled_ - track_pair_initial_dist.second);
      if (dist_travelled > params_.dist_travelled_together_to_initiate_leg_pair)
      {
        if (!track_pair.first->is_person_ && !track_pair.second->is_person_)
        {
          double new_pos_x = (track_pair.first->pos_x_ + track_pair.second->pos_x_) / 2;
          double new_pos_y = (track_pair.first->pos_y_ + track_pair.second->pos_y_) / 2;
          bool has_person_nearby = false;
          for (const auto& person_track : person_tracks)
          {
            if (!person_track->deleted_)
            {
              double dist = std::sqrt(std::pow(new_pos_x - person_track->pos_x_, 2) +
                                      std::pow(new_pos_y - person_track->pos_y_, 2));
              if (dist < 0.5)
              {
                has_person_nearby = true;
                break;
              }
            }
          }
          if (!has_person_nearby)
          {
            // Create a new person from this leg pair
            objects_tracked_.emplace_back(std::make_shared<ObjectTracked>(
                new_pos_x, new_pos_y, now, (track_pair.first->confidence_ + track_pair.second->confidence_) / 2, true));
            track_pair.first->deleted_ = true;
            track_pair.second->deleted_ = true;
            objects_tracked_.erase(std::remove(objects_tracked_.begin(), objects_tracked_.end(), track_pair.first),
                                   objects_tracked_.end());  // Using erase-remove idiom maybe too slow
            objects_tracked_.erase(std::remove(objects_tracked_.begin(), objects_tracked_.end(), track_pair.second),
                                   objects_tracked_.end());
            leg_pairs_to_delete.emplace(it);
            ROS_DEBUG_STREAM("Initialize a person track from a leg pair. ID: " << objects_tracked_.back()->id_num_);
          }
        }
        else if (track_pair.first->is_person_)
        {
          // Matched a tracked person to a tracked leg. Just delete the leg
          // and the person will hopefully be matched next iteration
          track_pair.second->deleted_ = true;
          objects_tracked_.erase(std::remove(objects_tracked_.begin(), objects_tracked_.end(), track_pair.second),
                                 objects_tracked_.end());
          leg_pairs_to_delete.emplace(it);
        }
        else 
        {
          // Matched a tracked person to a tracked leg. Just delete the leg
          // and the person will hopefully be matched next iteration
          track_pair.first->deleted_ = true;
          objects_tracked_.erase(std::remove(objects_tracked_.begin(), objects_tracked_.end(), track_pair.first),
                                 objects_tracked_.end());
          leg_pairs_to_delete.emplace(it);
        }
      }
    }
  }

  // Delete leg pairs set for deletion
  for (auto leg_pair : leg_pairs_to_delete)
  {
    potential_leg_pair_initial_dist_travelled_.erase((*leg_pair));
    potential_leg_pairs_.erase(leg_pair);
  }

  // Publish to rviz and /people_tracked_topic
  if (params_.publish_candidates)
  {
    publish_tracked_objects(now);
  }
  publish_tracked_people(now);

  ROS_DEBUG_STREAM("ObjectsTracked: " << objects_tracked_.size());
}

bool KalmanMultiTracker::checkDeletion(const std::shared_ptr<ObjectTracked>& track)
{
  double cov = track->filtered_state_covariances_cv_(0, 0) + track->var_obs_;
  if (track->is_person_)
  {
    if (track->times_seen_ >= 150)  // a mature track
    {
      if (track->not_seen_frames_ > 45)
      {
        ROS_DEBUG_STREAM(
            "Delete the mature person track because it has not been matched for 45 frames. ID: " << track->id_num_);
        return true;
      }
      else if (track->travelled_not_proper_ > 45)
      {
        ROS_DEBUG_STREAM("Delete the mature person track because it has not travelled properly for 45 frames. ID: "
                         << track->id_num_);
        return true;
      }
      if (cov > max_cov_)
      {
        ROS_DEBUG_STREAM("Delete the mature track because of high covariance. ID: " << track->id_num_);
        return true;
      }
      else
        return false;
    }
    else  // a young track
    {
      if (track->not_seen_frames_ > 30)
      {
        ROS_DEBUG_STREAM(
            "Delete the young person track because it has not been matched for 30 frames. ID: " << track->id_num_);
        return true;
      }
      else if (track->travelled_not_proper_ > 30)
      {
        ROS_DEBUG_STREAM("Delete the young person track because it has not travelled properly for 30 frames. ID: "
                         << track->id_num_);
        return true;
      }
      if (cov > max_cov_)
      {
        ROS_DEBUG_STREAM("Delete the young track because of high covariance. ID: " << track->id_num_);
        return true;
      }
      else
        return false;
    }
  }
  else  // candidate track
  {
    if (cov > max_cov_)
    {
      ROS_DEBUG_STREAM("Delete the candidate track because of high covariance. ID: " << track->id_num_);
      return true;
    }
    else if (track->not_seen_frames_ > 25)
    {
      ROS_DEBUG_STREAM(
          "Delete the candidate track because it has not been matched for 25 frames. ID: " << track->id_num_);
      return true;
    }
    else if (track->travelled_not_proper_ > 15)
    {
      ROS_DEBUG_STREAM(
          "Delete the candidate track because it has travelled improperly for 15 times. ID: " << track->id_num_);
      return true;
    }
    else if (track->single_dist_travelled_ > 0.5)
    {
      ROS_DEBUG_STREAM("Delete the candidate track because it moved too much in current frame. ID: " << track->id_num_);
      return true;
    }
    else
      return false;
  }
}

bool KalmanMultiTracker::checkInitialization(const std::shared_ptr<ObjectTracked>& track)
{
  double vel = std::sqrt(std::pow(track->vel_x_, 2) + std::pow(track->vel_y_, 2));

  if (params_.dynamic_mode)
  {
    if (track->times_seen_as_squat_consecutive_ > 10 && track->dist_to_init_pos_ > 0.5 &&
        track->dist_travelled_ > 0.8 && vel > 0.2 && vel < 2.0 && track->travelled_not_proper_ == 0)
    {
      if (track->confidence_ > 0.5)
      {
        return true;
      }
    }
    else
      return false;
  }
  else
  {
    if (track->times_seen_as_squat_consecutive_ > 3 && track->dist_to_init_pos_ > 0.3 && track->dist_travelled_ > 0.5 &&
        vel > 0.2 && vel < 2.0 && track->travelled_not_proper_ == 0)
    {
      if (track->confidence_ > 0.3)
      {
        return true;
      }
    }
    else
      return false;
  }
}

size_t ObjectTracked::new_leg_id_num_ = 1;
size_t DetectedCluster::new_cluster_id_num_ = 1;

}  // namespace laser_human_tracker
}