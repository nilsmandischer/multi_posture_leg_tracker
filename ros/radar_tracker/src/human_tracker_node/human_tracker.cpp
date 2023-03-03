#include "human_tracker.h"

namespace multi_posture_leg_tracker {

namespace radar_human_tracker
{
HumanTracker::HumanTracker() : nh_("~"), msg_num_(0), execution_time_(0.0), max_exec_time_(0.0), avg_exec_time_(0.0)
{
  prev_person_marker_id_ = 0;
  loadParameters();
  euclidian_dist_gate = 0.9;
  mahalanobis_dist_gate = 1.3;
  latest_scan_header_stamp_with_tf_available = ros::Time::now();

  // ROS publishers
  people_tracked_pub_ = nh_.advertise<rc_tracking_msgs::PersonArray>("/radar_tracked_people", 5);
  people_pose_pub_ = nh_.advertise<geometry_msgs::PoseArray>("/radar_tracked_people_pose_array", 5);
  marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/radar_tracked_people_marker", 5);

  // ROS subscribers
  detected_clusters_sub_ = nh_.subscribe("/radar_detected_clusters", 1, &HumanTracker::detectedClustersCallback, this);
}

void loadParameters()
{
  ros::NodeHandle nh_("~");
  // Get ROS params
  if (!nh_.getParam("fixed_frame", fixed_frame))
    std::cout << "Parameter Error: fixed_frame" << std::endl;
  if (!nh_.getParam("confidence_threshold_to_maintain_track", confidence_threshold_to_maintain_track))
    std::cout << "Parameter Error: confidence_threshold_to_maintain_track" << std::endl;
  if (!nh_.getParam("publish_people_frame", publish_people_frame))
    std::cout << "Parameter Error: publish_people_frame" << std::endl;
  if (!nh_.getParam("use_scan_header_stamp_for_tfs", use_scan_header_stamp_for_tfs))
    std::cout << "Parameter Error: use_scan_header_stamp_for_tfs" << std::endl;
  if (!nh_.getParam("para_p", para_p))
    std::cout << "Parameter Error: para_p" << std::endl;
  if (!nh_.getParam("std_obs", std_obs))
    std::cout << "Parameter Error: std_obs" << std::endl;
  if (!nh_.getParam("para_pos_q", para_pos_q))
    std::cout << "Parameter Error: para_pos_q" << std::endl;
  if (!nh_.getParam("para_vel_q", para_vel_q))
    std::cout << "Parameter Error: para_vel_q" << std::endl;
  if (!nh_.getParam("scan_frequency", scan_frequency))
    std::cout << "Parameter Error: scan_frequency" << std::endl;
  if (!nh_.getParam("distance_metric", distance_metric))
    std::cout << "Parameter Error: distance_metric" << std::endl;
  if (!nh_.getParam("matching_algorithm", matching_algorithm))
    std::cout << "Parameter Error: matching_algorithm" << std::endl;
  if (!nh_.getParam("dynamic_mode", dynamic_mode))
    std::cout << "Parameter Error: dynamic_mode" << std::endl;
  if (!nh_.getParam("publish_occluded", publish_occluded))
    std::cout << "Parameter Error: publish_occluded" << std::endl;

  delta_t = 1.0 / scan_frequency;
  std_noise_pos = para_pos_q / scan_frequency;
  std_noise_vel = para_vel_q / scan_frequency;
  var_obs = std::pow(std_obs + 0.4, 2);
}

void HumanTracker::detectedClustersCallback(const rc_tracking_msgs::LegArray& detected_clusters_msg)
{
  ros::Time now = detected_clusters_msg.header.stamp;

  std::vector<std::shared_ptr<DetectedCluster>> detected_clusters;
  for (const auto& cluster : detected_clusters_msg.legs)
  {
    detected_clusters.emplace_back(
        std::make_shared<DetectedCluster>(cluster.position.x, cluster.position.y, cluster.confidence));
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

  // Association between all tracks and detections
  std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>> person_track_detection_map;
  if (matching_algorithm == "GNN")
  {
    person_track_detection_map = match_detections_to_tracks_GNN(all_tracks, detected_clusters);
  }
  else if (matching_algorithm == "NN")
  {
    person_track_detection_map = match_detections_to_tracks_NN(all_tracks, detected_clusters);
  }
  else if (matching_algorithm == "greedy")
  {
    person_track_detection_map = match_detections_to_tracks_greedy_NN(all_tracks, detected_clusters);
  }
  else if (matching_algorithm == "MHT")
  {
    person_track_detection_map = match_detections_to_tracks_MHT(all_tracks, detected_clusters);
  }

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
    Eigen::Vector2d observation;
    std::shared_ptr<DetectedCluster> matched_detection(nullptr);
    if ((*it_track)->is_person_)  // person tracks
    {
      auto search_1 = person_track_detection_map.find(*it_track);
      auto search_2 = person_track_detection_map.find(duplicates.find(*it_track)->second);
      if (search_1 != person_track_detection_map.end() && search_2 != person_track_detection_map.end())
      {
        std::shared_ptr<DetectedCluster> md_1 = search_1->second;
        std::shared_ptr<DetectedCluster> md_2 = search_2->second;
        matched_detection = std::make_shared<DetectedCluster>(
            DetectedCluster((md_1->pos_x_ + md_2->pos_x_) / 2, (md_1->pos_y_ + md_2->pos_y_) / 2,
                            (md_1->confidence_ + md_2->confidence_) / 2));
      }
      else if (search_1 != person_track_detection_map.end())
      {
        matched_detection = search_1->second;
      }
      else if (search_2 != person_track_detection_map.end())
      {
        matched_detection = search_2->second;
      }
      else
      {
        matched_detection = nullptr;
      }
      if (matched_detection)
      {
        observation << matched_detection->pos_x_, matched_detection->pos_y_;
        (*it_track)->update(observation);

        (*it_track)->confidence_ = 0.7 * (*it_track)->confidence_ + 0.3 * matched_detection->confidence_;
        (*it_track)->times_seen_++;
        (*it_track)->times_seen_consecutive_++;
        (*it_track)->last_seen_ = now;
        (*it_track)->seen_in_current_scan_ = true;
        (*it_track)->not_seen_frames_ = 0;
      }
      else  // no matched detection
      {
        (*it_track)->times_seen_consecutive_ = 0;
        (*it_track)->seen_in_current_scan_ = false;
        (*it_track)->not_seen_frames_++;
      }

      (*it_track)->updateTraveledDist();

      // check Deletion Logic
      if (checkDeletion(*it_track))
      {
        (*it_track)->deleted_ = true;
        objects_tracked_.erase(it_track);
        it_track--;  // after erase it_track has already pointed to next component.
      }
    }
    else  // it's a candidate track
    {
      auto search = person_track_detection_map.find(*it_track);
      if (search != person_track_detection_map.end())
      {
        matched_detection = search->second;
        observation << matched_detection->pos_x_, matched_detection->pos_y_;
        (*it_track)->update(observation);

        (*it_track)->confidence_ = 0.7 * (*it_track)->confidence_ + 0.3 * matched_detection->confidence_;
        (*it_track)->times_seen_++;
        (*it_track)->times_seen_consecutive_++;
        (*it_track)->last_seen_ = now;
        (*it_track)->seen_in_current_scan_ = true;
        (*it_track)->not_seen_frames_ = 0;
      }
      else
      {
        (*it_track)->times_seen_consecutive_ = 0;
        (*it_track)->seen_in_current_scan_ = false;
        (*it_track)->not_seen_frames_++;
      }
      (*it_track)->updateTraveledDist();

      // check Deletion and Initialization Logic
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
            if (dist < 0.35)
            {
              has_person_nearby = true;
              break;
            }
          }
        }
        if (!has_person_nearby)
          (*it_track)->is_person_ = true;
      }
    }
  }

  // detections that have no matched tracks will be initialized as new tracks.
  for (const auto& detection : unmatched_detections)
  {
    objects_tracked_.emplace_back(std::make_shared<ObjectTracked>(detection->pos_x_, detection->pos_y_, now,
                                                                  detection->confidence_, false, detection->id_num_));
  }

  publish_tracked_people(now);
}

void ObjectTracked::update(Eigen::Vector2d observations)
{
  std::vector<double> near_part;

  /* update for KF, EKF, UKF */
  Tracker->updateKF(observations);
  /* update for PF */
  //    near_part = Tracker->updateWeights( observations);
  //    Tracker->resample();
  //    Tracker->x_[0] = near_part[0];
  //    Tracker->x_[1] = near_part[1];

  pos_x_ = Tracker->x_[0];
  pos_y_ = Tracker->x_[1];
  vel_x_ = Tracker->x_[2];
  vel_y_ = Tracker->x_[3];
}

void ObjectTracked::updateTraveledDist()
{
  double delta_dist_travelled = std::sqrt(std::pow(pos_x_ - last_pos_x_, 2) + std::pow(pos_y_ - last_pos_y_, 2));

  if (dynamic_mode)
  {
    if (delta_dist_travelled > 0.05)
    {
      dist_travelled_ += delta_dist_travelled;
    }
    actual_dist_travelled_ = delta_dist_travelled;
    if (delta_dist_travelled < 0.08 || delta_dist_travelled > 0.40)
    {
      travelled_not_proper_++;
    }
    else
    {
      travelled_not_proper_ = 0;
    }
  }
  else
  {
    if (delta_dist_travelled > 0.01)
    {
      dist_travelled_ += delta_dist_travelled;
    }
    actual_dist_travelled_ = delta_dist_travelled;
    if (delta_dist_travelled < 0.05 || delta_dist_travelled > 0.5)
    {
      travelled_not_proper_++;
    }
    else
    {
      travelled_not_proper_ = 0;
    }
  }

  dist_to_init_pos_ = std::sqrt(std::pow(pos_x_ - init_pos_x_, 2) + std::pow(pos_y_ - init_pos_y_, 2));

  last_pos_x_ = pos_x_;
  last_pos_y_ = pos_y_;
}

void ObjectTracked::predict()
{
  /** prediction for KF, EKF, UKF */
  Tracker->predictKF();
  /** prediction for PF */
  // Tracker->prediction(Tracker->x_[2], Tracker->x_[3]);
  pos_x_ = Tracker->x_[0];
  pos_y_ = Tracker->x_[1];
  vel_x_ = Tracker->x_[2];
  vel_y_ = Tracker->x_[3];
}

bool HumanTracker::checkDeletion(const std::shared_ptr<ObjectTracked>& track)
{
  if (dynamic_mode)
  {
    if (track->is_person_)
    {
      if (track->times_seen_ >= 50)  // a mature track
      {
        if (track->not_seen_frames_ > 10)
        {
          ROS_DEBUG_STREAM(
              "Delete the mature person track because it has not been matched for 10 frames. ID: " << track->id_num_);
          return true;
        }
        else if (track->confidence_ < confidence_threshold_to_maintain_track - 0.1)
        {
          ROS_DEBUG_STREAM("Delete the mature person track because of low confidence. ID: " << track->id_num_);
          return true;
        }
        else if (track->travelled_not_proper_ > 7)
        {
          ROS_DEBUG_STREAM("Delete the mature person track because it has not travelled properly for 8 frames. ID: "
                           << track->id_num_);
          return true;
        }
        else
          return false;
      }
      else  // a young track
      {
        if (track->not_seen_frames_ > 8)
        {
          ROS_DEBUG_STREAM(
              "Delete the young person track because it has not been matched for 8 frames. ID: " << track->id_num_);
          return true;
        }
        else if (track->confidence_ < confidence_threshold_to_maintain_track)
        {
          ROS_DEBUG_STREAM("Delete the young person track because of low confidence. ID: " << track->id_num_);
          return true;
        }
        else if (track->travelled_not_proper_ > 4)
        {
          ROS_DEBUG_STREAM("Delete the young person track because it has not travelled properly for 5 frames. ID: "
                           << track->id_num_);
          return true;
        }
        else
          return false;
      }
    }
    else  // candidate track
    {
      if (track->not_seen_frames_ > 8)
      {
        ROS_DEBUG_STREAM(
            "Delete the candidate track because it has not been matched for 8 frames. ID: " << track->id_num_);
        return true;
      }
      else if (track->actual_dist_travelled_ > 0.6)
      {
        ROS_DEBUG_STREAM(
            "Delete the candidate track because it moved too much in current frame. ID: " << track->id_num_);
        return true;
      }
      else
        return false;
    }
  }

  else
  {
    if (track->is_person_)
    {
      if (track->times_seen_ >= 50)  // a mature track
      {
        if (track->not_seen_frames_ > 20)
        {
          ROS_DEBUG_STREAM(
              "Delete the mature person track because it has not been matched for 20 frames. ID: " << track->id_num_);
          return true;
        }
        else if (track->confidence_ < confidence_threshold_to_maintain_track - 0.1)
        {
          ROS_DEBUG_STREAM("Delete the mature person track because of low confidence. ID: " << track->id_num_);
          return true;
        }
        else if (track->travelled_not_proper_ > 15)
        {
          ROS_DEBUG_STREAM("Delete the mature person track because it has not travelled properly for 15 frames. ID: "
                           << track->id_num_);
          return true;
        }
        else
          return false;
      }
      else  // a young track
      {
        if (track->not_seen_frames_ > 15)
        {
          ROS_DEBUG_STREAM(
              "Delete the young person track because it has not been matched for 15 frames. ID: " << track->id_num_);
          return true;
        }
        else if (track->confidence_ < confidence_threshold_to_maintain_track)
        {
          ROS_DEBUG_STREAM("Delete the young person track because of low confidence. ID: " << track->id_num_);
          return true;
        }
        else if (track->travelled_not_proper_ > 10)
        {
          ROS_DEBUG_STREAM("Delete the young person track because it has not travelled properly for 10 frames. ID: "
                           << track->id_num_);
          return true;
        }
        else
          return false;
      }
    }
    else  // candidate track
    {
      if (track->not_seen_frames_ > 8)
      {
        ROS_DEBUG_STREAM(
            "Delete the candidate track because it has not been matched for 8 frames. ID: " << track->id_num_);
        return true;
      }
      else if (track->actual_dist_travelled_ > 0.6)
      {
        ROS_DEBUG_STREAM(
            "Delete the candidate track because it moved too much in current frame. ID: " << track->id_num_);
        return true;
      }
      else
        return false;
    }
  }
}

bool HumanTracker::checkInitialization(const std::shared_ptr<ObjectTracked>& track)
{
  double vel = std::sqrt(std::pow(track->vel_x_, 2) + std::pow(track->vel_y_, 2));
  if (dynamic_mode)
  {
    if (track->times_seen_consecutive_ > 8 && track->dist_to_init_pos_ > 0.8 && track->dist_travelled_ > 1.0 &&
        track->travelled_not_proper_ == 0 && vel > 0.3 && vel < 2.0)
    {
      if (track->confidence_ > 0.5)
      {
        ROS_DEBUG_STREAM("Initialize the candidate as a person track. ID: " << track->id_num_);
        return true;
      }
    }
    else
      return false;
  }
  else  // dynamic_mode == false
  {
    if (track->times_seen_consecutive_ > 2 && track->dist_to_init_pos_ > 0.5 && track->dist_travelled_ > 0.7 &&
        track->travelled_not_proper_ == 0 && vel > 0.1 && vel < 2.0)
    {
      if (track->confidence_ > 0.3)
      {
        ROS_DEBUG_STREAM("Initialize the candidate as a person track. ID: " << track->id_num_);
        return true;
      }
    }
    else
      return false;
  }
}

ObjectTracked::~ObjectTracked(void)
{
}

ObjectTracked::ObjectTracked(double x, double y, ros::Time now, double confidence, bool is_person, int det_num)
  : is_person_(is_person)
  , last_seen_(now)
  , times_seen_(1)
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
  , times_seen_consecutive_(1)
  , color_{ static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX),
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX),
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) }
  , id_num_(new_leg_id_num_)
  , id_det_(det_num)
  , deleted_(false)
  , dist_travelled_(0.0)
  , actual_dist_travelled_(0.0)
  , dist_to_init_pos_(0.0)
  , travelled_not_proper_(0)
{
  new_leg_id_num_++;
  Eigen::Vector4d x_;
  x_ << x, y, 0, 0;
  //  Tracker = new KalmanFilter(delta_t, std_noise_pos, std_noise_vel, std_obs, para_p);
  // Tracker = new ExtendedKalmanFilter(delta_t, std_noise_pos, std_noise_vel, std_obs, para_p);
  // Tracker = new UnscentedKalmanFilter(delta_t, std_noise_pos, std_noise_vel, std_obs, para_p);
  Tracker = new KalmanFilterSH(delta_t, std_noise_pos, std_noise_vel, std_obs, para_p);
  // Tracker = new ParticleFilter(x_, delta_t, std_noise_pos, std_noise_vel, std_obs);

  Tracker->initializeKF(x_);
  Tracker->predictKF();
  /* for PF */
  // Tracker->prediction(x_[2], x_[3]);
}

ObjectTracked::ObjectTracked(const ObjectTracked& obj)
  : Tracker(obj.Tracker)
  , is_person_(obj.is_person_)
  , last_seen_(obj.last_seen_)
  , times_seen_(obj.times_seen_)
  , var_obs_(obj.var_obs_)
  , pos_x_(obj.pos_x_)
  , pos_y_(obj.pos_y_)
  , vel_x_(obj.vel_x_)
  , vel_y_(obj.vel_y_)
  , last_pos_x_(obj.last_pos_x_)
  , last_pos_y_(obj.last_pos_y_)
  , init_pos_x_(obj.init_pos_x_)
  , init_pos_y_(obj.init_pos_y_)
  , id_det_(obj.id_det_)
  , confidence_(obj.confidence_)
  , times_seen_consecutive_(obj.times_seen_consecutive_)
  , seen_in_current_scan_(obj.seen_in_current_scan_)
  , not_seen_frames_(obj.not_seen_frames_)
  , color_{ obj.color_[0], obj.color_[1], obj.color_[2] }
  , id_num_(obj.id_num_)
  , deleted_(obj.deleted_)
  , dist_travelled_(obj.dist_travelled_)
  , actual_dist_travelled_(obj.actual_dist_travelled_)
  , dist_to_init_pos_(obj.dist_to_init_pos_)
  , travelled_not_proper_(obj.travelled_not_proper_)
{
  Tracker->predictKF();
  /* for PF */
  // Tracker->prediction(vel_x_,vel_y_);
}

void HumanTracker::publish_tracked_people(ros::Time now)
{
  rc_tracking_msgs::PersonArray people_tracked_msg;
  people_tracked_msg.header.stamp = now;
  people_tracked_msg.header.frame_id = publish_people_frame;

  geometry_msgs::PoseArray people_pose_array;
  people_pose_array.header = people_tracked_msg.header;

  // Make sure we can get the required transform first:
  bool transform_available;
  ros::Time tf_time;
  if (use_scan_header_stamp_for_tfs)
  {
    tf_time = now;
    try
    {
      listener_.waitForTransform(publish_people_frame, fixed_frame, tf_time, ros::Duration(1.0));
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
    transform_available = listener_.canTransform(publish_people_frame, fixed_frame, tf_time);
  }

  int marker_id = 0;
  visualization_msgs::MarkerArray markers;
  if (!transform_available)
  {
    ROS_WARN("Radar Tracker: tf not available. Not publishing tracked people");
  }
  else
  {
    for (auto person : objects_tracked_)
    {
      if (person->is_person_)
      {
        if (publish_occluded || person->seen_in_current_scan_)
        {
          // Only publish people who have been seen in current scan, unless we
          // want to publish occluded people
          // Get position in the <publish_people_frame> frame
          geometry_msgs::PointStamped ps;
          ps.header.frame_id = fixed_frame;
          ps.header.stamp = tf_time;
          ps.point.x = person->pos_x_;
          ps.point.y = person->pos_y_;
          try
          {
            listener_.transformPoint(publish_people_frame, ps, ps);
          }
          catch (...)
          {
            ROS_WARN_STREAM("Not publishing tracked people due to no transform from fixed_frame: "
                            << fixed_frame << "to publish_people_frame: " << publish_people_frame);
            continue;
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
          marker.header.frame_id = publish_people_frame;
          marker.header.stamp = now;
          marker.ns = "radar_tracked_people";
          marker.color.r = 0;
          marker.color.g = 0;
          marker.color.b = 1;
          marker.color.a = 1;
          marker.pose.position.x = ps.point.x;
          marker.pose.position.y = ps.point.y;
          marker.id = marker_id;
          marker_id++;
          marker.type = visualization_msgs::Marker::CYLINDER;
          marker.scale.x = 0.23;
          marker.scale.y = 0.23;
          marker.scale.z = 1.2;
          marker.pose.position.z = 0.8;
          markers.markers.push_back(marker);

          // Sphere for head shape
          marker.type = visualization_msgs::Marker::SPHERE;
          marker.scale.x = 0.2;
          marker.scale.y = 0.2;
          marker.scale.z = 0.2;
          marker.pose.position.z = 1.55;
          marker.id = marker_id;
          marker_id++;
          markers.markers.push_back(marker);

          // Text showing person's ID number
          marker.color.r = 0.0;
          marker.color.g = 0.0;
          marker.color.b = 0.0;
          marker.color.a = 1.0;
          marker.id = marker_id;
          marker_id++;
          marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
          marker.text = std::to_string(person->id_num_);
          marker.scale.z = 0.4;
          marker.pose.position.z = 1.85;
          markers.markers.push_back(marker);
        }
      }
    }
  }
  // Delete remaining markers from the stage before
  for (size_t m_id = marker_id; m_id < prev_person_marker_id_; m_id++)
  {
    visualization_msgs::Marker marker;
    marker.header.stamp = now;
    marker.header.frame_id = publish_people_frame;
    marker.ns = "radar_tracked_people";
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

std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>>
HumanTracker::match_detections_to_tracks_GNN(const std::vector<std::shared_ptr<ObjectTracked>>& objects_tracked,
                                             const std::vector<std::shared_ptr<DetectedCluster>>& objects_detected)
{
  std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>> matched_tracks{};

  // Populate match_dist matrix of mahalanobis_dist/euclidian_dist between every detection and
  // every track
  std::vector<std::vector<double>> match_dist;  // matrix of probability of matching between all people and
                                                // all detections
  std::vector<std::shared_ptr<DetectedCluster>> eligable_detections;  // Only include detections in match_dist matrix if
                                                                      // they're in range of at least one track to speed
                                                                      // up munkres
  match_dist.reserve(objects_detected.size());
  eligable_detections.reserve(objects_detected.size());

  for (const auto& detect : objects_detected)
  {
    bool at_least_one_track_in_range = false;
    std::vector<double> new_row;
    new_row.reserve(objects_tracked.size());
    for (const auto& track : objects_tracked)
    {
      double cost;
      if (distance_metric == "mahalanobis")
      {
        // Use mahalanobis dist to do matching
        double mahalanobis_dist = std::sqrt(
            (std::pow(detect->pos_x_ - track->pos_x_, 2) + std::pow(detect->pos_y_ - track->pos_y_, 2)) / var_obs);
        if (mahalanobis_dist < mahalanobis_dist_gate)
        {
          cost = mahalanobis_dist;
          at_least_one_track_in_range = true;
        }
        else
        {
          cost = kMax_cost_;
        }
      }
      else if (distance_metric == "euclidian")
      {
        // Use euclidian dist to do matching
        double eulclidian_dist =
            std::sqrt((std::pow(detect->pos_x_ - track->pos_x_, 2) + std::pow(detect->pos_y_ - track->pos_y_, 2)));
        if (eulclidian_dist < euclidian_dist_gate)
        {
          cost = eulclidian_dist;
          at_least_one_track_in_range = true;
        }
        else
        {
          cost = kMax_cost_;
        }
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
        if (match_dist.at(elig_detect_idx).at(assignment.at(elig_detect_idx)) < euclidian_dist_gate)
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

std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>>
HumanTracker::match_detections_to_tracks_NN(const std::vector<std::shared_ptr<ObjectTracked>>& objects_tracked,
                                            const std::vector<std::shared_ptr<DetectedCluster>>& objects_detected)
{
  std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>> matched_tracks;
  std::unordered_map<std::size_t, std::pair<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>>>
      potential_pairs_;
  for (const auto& detect : objects_detected)
  {
    for (const auto& track : objects_tracked)
    {
      if (distance_metric == "mahalanobis")
      {
        // Use mahalanobis dist to do matching
        double mahalanobis_dist = std::sqrt(
            (std::pow(detect->pos_x_ - track->pos_x_, 2) + std::pow(detect->pos_y_ - track->pos_y_, 2)) / var_obs);
        if (mahalanobis_dist < mahalanobis_dist_gate)
        {
          potential_pairs_.emplace(track->id_num_, std::make_pair(track, detect));
        }
      }
      else if (distance_metric == "euclidian")
      {
        // Use euclidian dist to do matching
        double euclidian_dist =
            std::sqrt((std::pow(detect->pos_x_ - track->pos_x_, 2) + std::pow(detect->pos_y_ - track->pos_y_, 2)));
        if (euclidian_dist < euclidian_dist_gate)
        {
          potential_pairs_.emplace(track->id_num_, std::make_pair(track, detect));
        }
      }
    }
  }

  std::shared_ptr<ObjectTracked> temp_pairing_track;
  std::shared_ptr<DetectedCluster> temp_pairing_detect;
  std::vector<int> obs_num;

  for (const auto& track : objects_tracked)
  {
    auto track_id = potential_pairs_.equal_range(track->id_num_);
    float best_matching_gate = 0;
    for (auto it = track_id.first; it != track_id.second; it++)
    {
      if (distance_metric == "mahalanobis")
      {
        float dist = std::sqrt(std::pow((it->second.first->pos_x_ - it->second.second->pos_x_), 2) +
                               (std::pow((it->second.first->pos_y_ - it->second.second->pos_y_), 2)) / var_obs);
        if (dist < mahalanobis_dist_gate)
        {
          best_matching_gate = dist;
          temp_pairing_detect = it->second.second;
          temp_pairing_track = it->second.first;
        }
      }
      else if (distance_metric == "euclidian")
      {
        float dist = std::sqrt(std::pow((it->second.first->pos_x_ - it->second.second->pos_x_), 2) +
                               (std::pow((it->second.first->pos_y_ - it->second.second->pos_y_), 2)));
        if (dist < euclidian_dist_gate)
        {
          best_matching_gate = dist;
          temp_pairing_detect = it->second.second;
          temp_pairing_track = it->second.first;
        }
      }
    }
    std::vector<int>::iterator it_obs = std::find(obs_num.begin(), obs_num.end(), temp_pairing_detect->id_num_);

    if (temp_pairing_detect && (it_obs == obs_num.end()))
    {
      temp_pairing_detect->is_matched_ = true;
      matched_tracks.emplace(temp_pairing_track, temp_pairing_detect);
      obs_num.emplace_back(temp_pairing_detect->id_num_);
    }
  }
  return matched_tracks;
}

std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>>
HumanTracker::match_detections_to_tracks_greedy_NN(
    const std::vector<std::shared_ptr<ObjectTracked>>& objects_tracked,
    const std::vector<std::shared_ptr<DetectedCluster>>& objects_detected)
{
  std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>> matched_tracks{};

  // Populate match_dist matrix of mahalanobis_dist between every detection and
  // every track
  Eigen::MatrixXd match_dist = Eigen::MatrixXd::Constant(objects_tracked.size(), objects_detected.size(), BIG_COST);
  if (objects_tracked.size() > 0)
  {
    for (size_t detect = 0; detect < objects_detected.size(); detect++)
    {
      std::vector<double> new_row;
      new_row.reserve(objects_tracked.size());
      for (size_t track = 0; track < objects_tracked.size(); track++)
      {
        if (distance_metric == "mahalanobis")
        {
          // Use mahalanobis dist to do matching
          double mahalanobis_dist =
              std::sqrt((std::pow(objects_detected.at(detect)->pos_x_ - (objects_tracked.at(track)->pos_x_), 2) +
                         std::pow(objects_detected.at(detect)->pos_y_ - (objects_tracked.at(track)->pos_y_), 2)) /
                        var_obs);
          if (mahalanobis_dist < mahalanobis_dist_gate)
          {
            match_dist(track, detect) = mahalanobis_dist;
          }
        }
        else if (distance_metric == "euclidian")
        {
          // Use euclidian dist to do matching
          double euclidian_dist =
              std::sqrt((std::pow(objects_detected.at(detect)->pos_x_ - (objects_tracked.at(track)->pos_x_), 2) +
                         std::pow(objects_detected.at(detect)->pos_y_ - (objects_tracked.at(track)->pos_y_), 2)));
          if (euclidian_dist < euclidian_dist_gate)
          {
            match_dist(track, detect) = euclidian_dist;
          }
        }
      }
    }

    // Step 2: find minimum in cost Matrix until there is no valid pairing anymore
    Eigen::VectorXd observationBIG = Eigen::VectorXd::Constant(objects_detected.size(), VERY_BIG_COST);
    Eigen::VectorXd tracksBIG = Eigen::VectorXd::Constant(objects_tracked.size(), VERY_BIG_COST);

    double currentCost = 0;
    int i, j;
    if (objects_detected.size() > 0)
    {
      do
      {
        currentCost = match_dist.minCoeff(&i, &j);
        if (currentCost < BIG_COST)
        {
          objects_detected.at(j)->is_matched_ = true;
          matched_tracks.emplace(objects_tracked.at(i), objects_detected.at(j));
          match_dist.row(i) = observationBIG;
          match_dist.col(j) = tracksBIG;
        }

      } while (currentCost < BIG_COST);
    }
  }
  return matched_tracks;
}

std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>>
HumanTracker::match_detections_to_tracks_MHT(const std::vector<std::shared_ptr<ObjectTracked>>& objects_tracked,
                                             const std::vector<std::shared_ptr<DetectedCluster>>& objects_detected)
{
  std::unordered_map<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>> matched_tracks{};
  std::unordered_map<std::size_t, std::pair<std::shared_ptr<ObjectTracked>, std::shared_ptr<DetectedCluster>>>
      potential_pairs_;
  std::vector<double> dist;
  std::vector<int> obj_id;
  int i = 0;
  std::vector<std::shared_ptr<ObjectTracked>> objects_unique;
  for (const auto track : objects_tracked)
  {
    std::vector<int>::iterator it = std::find(obj_id.begin(), obj_id.end(), track->id_num_);
    if (it != obj_id.end())
    {
      continue;
    }
    else
    {
      objects_unique.push_back(track);
    }
    obj_id.push_back(track->id_num_);
    i++;
  }

  // 1. Gating
  for (const auto& detect : objects_detected)
  {
    for (const auto& track : objects_unique)
    {
      if (distance_metric == "mahalanobis")
      {
        // Use mahalanobis dist to do matching
        double mahalanobis_dist = std::sqrt(
            (std::pow(detect->pos_x_ - track->pos_x_, 2) + std::pow(detect->pos_y_ - track->pos_y_, 2)) / var_obs);
        if (mahalanobis_dist < mahalanobis_dist_gate)
        {
          potential_pairs_.emplace(track->id_det_, std::make_pair(track, detect));
          dist.push_back(mahalanobis_dist);
        }
      }
      else if (distance_metric == "euclidian")
      {
        // Use euclidian dist to do matching
        double euclidian_dist =
            std::sqrt((std::pow(detect->pos_x_ - track->pos_x_, 2) + std::pow(detect->pos_y_ - track->pos_y_, 2)));
        if (euclidian_dist < euclidian_dist_gate)
        {
          potential_pairs_.emplace(track->id_det_, std::make_pair(track, detect));
          dist.push_back(euclidian_dist);
        }
      }
    }
  }
  // 2. Track Formation
  for (auto det : objects_detected)
  {
    History h;
    h.first_id = det->id_num_;
    h.id = det->id_num_;
    track_history.push_back(h);
  }

  // create history
  for (auto det : objects_detected)
  {
    int it = 0;
    for (auto pot : potential_pairs_)
    {
      if (det->id_num_ == pot.second.second->id_num_)
      {
        for (auto& his : track_history)
        {
          if (his.id == pot.second.first->id_det_)
          {
            double dx = det->pos_x_ - pot.second.first->pos_x_;
            double dy = det->pos_y_ - pot.second.first->pos_y_;
            his.first_id = det->id_num_;
            his.probability *= 1.0 / (2 * PI * std_obs * std_obs) * exp(-dx * dx / (2 * std_obs * std_obs)) *
                               exp(-dy * dy / (2 * std_obs * std_obs));
            his.dist = dist[it];
          }
        }
      }
      it++;
    }
  }

  // Add Track with no matching
  for (auto his : track_history)
  {
    if (!his.new_track && his.first_id != 0)
    {
      History h;
      h.id = his.id;
      h.first_id = 0;
      h.second_id = his.second_id;
      h.third_id = his.third_id;
      h.fourth_id = his.fourth_id;
      h.fifth_id = his.fifth_id;
      track_history.push_back(h);
    }
  }

  // 3. Reduction of Tracks
  // Reduction due to low probability
  std::vector<History> temp;
  for (auto his : track_history)
  {
    if (his.new_track)
    {
      temp.push_back(his);
    }
    if (!his.new_track && his.probability >= 0.5)
    {
      temp.push_back(his);
    }
  }

  track_history.clear();
  std::vector<History>::iterator it;
  it = track_history.begin();
  track_history.insert(it, temp.begin(), temp.end());
  temp.clear();

  // Reduction due to similarities - track merging --- actual only one digit can be different for merging
  std::vector<History> track_history_temp;
  std::vector<int> objs_id;
  for (auto o_t : objects_unique)
  {
    objs_id.push_back(o_t->id_det_);
  }

  // Reduction if tracks in track_history which aren't tracked anymore
  for (auto his = track_history.begin(); his != track_history.end(); ++his)
  {
    std::vector<int>::iterator it = std::find(objs_id.begin(), objs_id.end(), his->id);
    if (it != objs_id.end())
    {
      continue;
    }
    else
    {
      if ((his->fifth_id == 0) && (his->fourth_id == 0) && (his->third_id == 0) && (his->first_id == 0) &&
          (his->second_id == 0))
      {
        track_history_temp.push_back(*his);
      }
    }
  }

  for (auto his = track_history.begin(); his != track_history.end(); ++his)
  {
    for (auto track_his = track_history.begin(); track_his != his; ++track_his)
    {
      if (his->id == track_his->id)
      {
        if ((his->fifth_id == track_his->fifth_id) && (his->fourth_id == track_his->fourth_id) &&
            (his->third_id == track_his->third_id) && (his->second_id == track_his->second_id))
        {
          if (his->first_id != 0)
          {
            track_history_temp.push_back(*track_his);
            break;
          }
          else
          {
            track_history_temp.push_back(*his);
            break;
          }
        }
        if ((his->fifth_id == track_his->fifth_id) && (his->fourth_id == track_his->fourth_id) &&
            (his->third_id == track_his->third_id) && (his->first_id == track_his->first_id))
        {
          if (his->second_id != 0)
          {
            track_history_temp.push_back(*track_his);
            break;
          }
          else
          {
            track_history_temp.push_back(*his);
            break;
          }
        }
      }
    }
  }

  for (auto his = track_history_temp.begin(); his != track_history_temp.end(); ++his)
  {
    for (auto track_his = track_history.begin(); track_his != track_history.end() && track_his != his; ++track_his)
    {
      if ((his->id == track_his->id) && (his->fifth_id == track_his->fifth_id) &&
          (his->fourth_id == track_his->fourth_id) && (his->third_id == track_his->third_id) &&
          (his->second_id == track_his->second_id) && (his->first_id == track_his->first_id))
      {
        track_history.erase(track_his);
        break;
      }
    }
  }
  track_history_temp.clear();

  // 4. N-Scan pruning

  for (auto& track1 : track_history)
  {
    track1.fifth_id = track1.fourth_id;
    track1.fourth_id = track1.third_id;
    track1.third_id = track1.second_id;
    track1.second_id = track1.first_id;
    track1.first_id = std::nan("0");
    track1.new_track = false;
  }

  // 5. Best Hypothesis

  std::pair<int, int> id;
  std::pair<int, double> id_di;

  std::vector<std::pair<double, std::pair<int, int>>> id_connections;

  std::vector<std::pair<int, double>> id_dist;
  for (auto tra : objects_unique)
  {
    double proba = 0;
    id = std::make_pair(99999999, 99999999);
    for (auto his : track_history)
    {
      if ((his.id == tra->id_det_) && his.probability >= proba)
      {
        id = std::make_pair(tra->id_num_, his.second_id);
        proba = his.dist;
        id_di = std::make_pair(tra->id_num_, proba);
      }
    }
    if (id.first != 99999999)
    {
      std::pair<double, std::pair<int, int>> temp;
      temp = std::make_pair(proba, id);
      id_connections.push_back(temp);
    }
  }
  std::vector<int> ids;
  int f = 0;
  for (auto idc : id_connections)
  {
    if (idc.second.second == 0)
    {
      ids.push_back(f);
    }
    f++;
  }
  for (int i = ids.size(); i != 0; i--)
  {
    id_connections.erase(id_connections.begin() + ids[i - 1]);
  }
  std::sort(id_connections.begin(), id_connections.end());
  std::vector<std::pair<double, std::pair<int, int>>> id_con;
  std::vector<int> detection_id;
  for (auto idc : id_connections)
  {
    std::vector<int>::iterator det_ids = std::find(detection_id.begin(), detection_id.end(), idc.second.second);
    if (det_ids != detection_id.end())
    {
      continue;
    }
    else
    {
      id_con.push_back(idc);
      detection_id.push_back(idc.second.second);
    }
  }
  while (id_con.size() > 0)
  {
    for (auto tra : objects_unique)
    {
      for (auto det : objects_detected)
      {
        if ((tra->id_num_ == id_con[0].second.first) && (det->id_num_ == id_con[0].second.second))
        {
          det->is_matched_ = true;
          matched_tracks.emplace(tra, det);
          id_con.erase(id_con.begin());
        }
      }
    }
  }
  return matched_tracks;
}

size_t ObjectTracked::new_leg_id_num_ = 1;
size_t DetectedCluster::new_cluster_id_num_ = 1;

}  

}  // namespace radar_human_tracker
