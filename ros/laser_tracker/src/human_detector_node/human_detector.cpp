#include "human_detector.h"

namespace multi_posture_leg_tracker
{

namespace laser_human_tracker
{
HumanDetector::HumanDetector(ros::NodeHandle nh)
  : nh_("~"), scan_num_(0), num_prev_markers_published_(0), execution_time_(0), avg_exec_time_(0), max_exec_time_(0)
{
  // Get ROS parameters
  std::string scan_topic;
  nh_.param("scan_topic", scan_topic, std::string("scan"));
  nh_.param("fixed_frame", fixed_frame_, std::string("odom"));
  nh_.param("detection_threshold", detection_threshold_, -1.0);
  nh_.param("cluster_dist_euclid", cluster_dist_euclid_, 0.13);
  nh_.param("min_points_per_cluster", min_points_per_cluster_, 3);
  nh_.param("max_detect_distance", max_detect_distance_, 10.0);
  nh_.param("use_scan_header_stamp_for_tfs", use_scan_header_stamp_for_tfs_, false);
  nh_.param("publish_background", publish_background_, false);
  nh_.param("visualize_contour", visualize_contour_, false);
  nh_.param("max_detected_clusters", max_detected_clusters_, -1);
  nh_.param("classifier_type", classifier_type_, std::string("opencv_random_forest"));
  std::string model_file;
  if (!nh_.getParam("model_file", model_file))
    ROS_ERROR("ERROR! Could not get model filename");
  std::string detected_clusters_topic;
  std::string detected_clusters_marker_topic;
  nh_.param("detected_clusters_topic", detected_clusters_topic, std::string("laser_detected_clusters"));
  nh_.param("detected_clusters_marker_topic", detected_clusters_marker_topic,
            std::string("laser_detected_clusters_marker"));

  // Print back
  ROS_INFO("scan_topic: %s", scan_topic.c_str());
  ROS_INFO("fixed_frame: %s", fixed_frame_.c_str());
  ROS_INFO("detection_threshold: %.2f", detection_threshold_);
  ROS_INFO("cluster_dist_euclid: %.2f", cluster_dist_euclid_);
  ROS_INFO("min_points_per_cluster: %d", min_points_per_cluster_);
  ROS_INFO("max_detect_distance: %.2f", max_detect_distance_);
  ROS_INFO("use_scan_header_stamp_for_tfs: %d", use_scan_header_stamp_for_tfs_);
  ROS_INFO("max_detected_clusters: %d", max_detected_clusters_);
  ROS_INFO("classifier_type: %s", classifier_type_.c_str());
  ROS_INFO("model_file: %s", model_file.c_str());

  // Initialize and load classifier
  if (classifier_type_ == "opencv_random_forest")
    classifier_ = std::make_shared<OpenCVRandomForestClassifier>(nh);
  else if (classifier_type_ == "opencv_adaboost")
    classifier_ = std::make_shared<OpenCVAdaBoostClassifier>(nh);
  else if (classifier_type_ == "mlpack_random_forest")
    classifier_ = std::make_shared<MlpackRandomForestClassifier>(nh);
  else if (classifier_type_ == "mlpack_adaboost")
    classifier_ = std::make_shared<MlpackAdaBoostClassifier>(nh);
  else
  {
    ROS_ERROR("The classifier type should be opencv_random_forest OR opencv_adaboost OR mlpack_random_forest OR "
              "mlpack_adaboost! No such type: %s",
              classifier_type_.c_str());
    exit(1);
  }

  classifier_->loadModel(model_file);
  feat_count_ = classifier_->getFeatureSetSize();

  switch (feat_count_)
  {
    case FEATURE_SET_0:
      feature_set_size_ = 0;
      break;
    case FEATURE_SET_1:
      feature_set_size_ = 1;
      break;
    case FEATURE_SET_2:
      feature_set_size_ = 2;
      break;
    case FEATURE_SET_3:
      feature_set_size_ = 3;
      break;
    case FEATURE_SET_4:
      feature_set_size_ = 4;
      break;
    default:
      ROS_ERROR("There is no such feature set size!");
  }
  ROS_INFO("The feature set size is %d", feature_set_size_);

  // ROS subscribers + publishers
  scan_sub_ = nh_.subscribe(scan_topic, 1, &laser_human_tracker::HumanDetector::laserCallback, this);
  markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(detected_clusters_marker_topic, 5);
  detected_clusters_pub_ = nh_.advertise<rc_tracking_msgs::LegArray>(detected_clusters_topic, 5);
}

void HumanDetector::laserCallback(const sensor_msgs::LaserScan::ConstPtr& scan)
{
  scan_num_++;
  ros::WallTime start, end;
  start = ros::WallTime::now();

  laser_processor::ScanProcessor processor(*scan);
  processor.splitConnected(cluster_dist_euclid_);
  processor.removeLessThan(min_points_per_cluster_);

  rc_tracking_msgs::LegArray detected_clusters;
  detected_clusters.header.frame_id = scan->header.frame_id;
  detected_clusters.header.stamp = scan->header.stamp;

  visualization_msgs::MarkerArray rviz_markers;
  int id_num = 0;

  // Find out the time that should be used for tfs
  bool transform_available;
  ros::Time tf_time;
  // Use time from scan header
  if (use_scan_header_stamp_for_tfs_)
  {
    tf_time = scan->header.stamp;

    try
    {
      tfl_.waitForTransform(fixed_frame_, scan->header.frame_id, tf_time, ros::Duration(1.0));
      transform_available = tfl_.canTransform(fixed_frame_, scan->header.frame_id, tf_time);
    }
    catch (tf::TransformException ex)
    {
      ROS_WARN("Laser Detector: No tf available");
      transform_available = false;
    }
  }
  else
  {
    // Otherwise just use the latest tf available
    tf_time = ros::Time(0);
    transform_available = tfl_.canTransform(fixed_frame_, scan->header.frame_id, tf_time);
  }

  // Store all processed clusters in a set ordered according to their relative
  // distance to the laser scanner
  std::set<rc_tracking_msgs::Leg, CompareClusters> cluster_set;

  if (!transform_available)
  {
    ROS_WARN("Not publishing detected clusters because no tf was available");
  }
  else  // transform_available
  {
    // Iterate through all clusters
    for (std::list<laser_processor::SampleSet*>::iterator cluster = processor.getClusters().begin();
         cluster != processor.getClusters().end(); cluster++)
    {
      // Get position of cluster in laser frame
      tf::Stamped<tf::Point> position((*cluster)->getPosition(), tf_time, scan->header.frame_id);
      float rel_dist = pow(position[0] * position[0] + position[1] * position[1], 1. / 2.);

      // Only consider clusters within max_distance.
      if (rel_dist < max_detect_distance_)
      {
        // Classify cluster using random forest classifier
        std::vector<float> f = cf_.calcClusterFeatures(*cluster, *scan, feature_set_size_);

        // Run classifier to determine cluster's label and confidence level of human
        float label;
        float probability_of_human;
        classifier_->classifyFeatureVector(f, label, probability_of_human);

        // Consider only clusters that have a confidence greater than
        // detection_threshold_
        if (probability_of_human > detection_threshold_)
        {
          // Transform cluster position to fixed frame
          // This should always be succesful because we've checked earlier if
          // a tf was available
          bool transform_successful_2;
          try
          {
            tfl_.transformPoint(fixed_frame_, position, position);
            transform_successful_2 = true;
          }
          catch (tf::TransformException ex)
          {
            ROS_ERROR("%s", ex.what());
            transform_successful_2 = false;
          }

          if (transform_successful_2)
          {
            // Add detected cluster to set of detected clusters, along
            // with its relative position to the laser scanner
            rc_tracking_msgs::Leg new_cluster;
            new_cluster.position.x = position[0];
            new_cluster.position.y = position[1];
            new_cluster.confidence = probability_of_human;
            if (std::abs(label - 1) <= FLT_EPSILON)
            {
              new_cluster.label = rc_tracking_msgs::Leg::LABEL_LEG;
              cluster_set.insert(new_cluster);
            }
            else if (std::abs(label - 2) <= FLT_EPSILON)
            {
              new_cluster.label = rc_tracking_msgs::Leg::LABEL_SQUAT;
              cluster_set.insert(new_cluster);
            }
            else
            {
              new_cluster.label = rc_tracking_msgs::Leg::LABEL_BKGD;
              if (publish_background_)
              {
                cluster_set.insert(new_cluster);
              }
            }
          }
        }
      }

      // visualize the extracted closed contour in rviz
      if (visualize_contour_)
      {
        visualization_msgs::Marker polygon_msg;
        polygon_msg.header.frame_id = fixed_frame_;
        polygon_msg.header.stamp = scan->header.stamp;
        polygon_msg.ns = "laser_detected_clusters";
        polygon_msg.type = visualization_msgs::Marker::LINE_STRIP;
        polygon_msg.scale.x = 0.01;
        polygon_msg.scale.y = 0.01;
        polygon_msg.scale.z = 0.01;
        polygon_msg.color.g = 1.0;
        polygon_msg.color.a = 0.8;
        id_num++;
        polygon_msg.id = id_num;
        for (laser_processor::SampleSet::iterator i = (*cluster)->begin(); i != (*cluster)->end(); i++)
        {
          geometry_msgs::Point p;
          p.x = (*i)->x;
          p.y = (*i)->y;
          p.z = 0;
          polygon_msg.points.push_back(p);
        }
        rviz_markers.markers.push_back(polygon_msg);
      }
    }
  }

  // Publish detected clusters to laser_detected_clusters and to rviz
  // They are ordered from closest to the laser scanner to furthest
  int clusters_published_counter = 0;
  visualization_msgs::Marker m;
  visualization_msgs::Marker delete_m;
  visualization_msgs::Marker confidence_m;

  for (std::set<rc_tracking_msgs::Leg>::iterator it = cluster_set.begin(); it != cluster_set.end(); ++it)
  {
    // Publish to laser_detected_clusters topic
    rc_tracking_msgs::Leg cluster = *it;
    detected_clusters.legs.push_back(cluster);
    clusters_published_counter++;

    // Publish marker to rviz
    id_num++;
    m.header.stamp = scan->header.stamp;
    m.header.frame_id = fixed_frame_;
    m.ns = "laser_detected_clusters";
    m.id = id_num;
    m.pose.position.x = cluster.position.x;
    m.pose.position.y = cluster.position.y;
    m.pose.position.z = 0.0;
    m.scale.x = 0.1;  // 0.13
    m.scale.y = 0.1;
    m.scale.z = 0.1;
    m.color.a = 1;
    m.color.r = 0;
    m.color.g = cluster.confidence;
    m.color.b = cluster.confidence;

    switch (cluster.label)
    {
      case rc_tracking_msgs::Leg::LABEL_LEG:
        m.type = visualization_msgs::Marker::SPHERE;
        break;
      case rc_tracking_msgs::Leg::LABEL_SQUAT:
        m.type = visualization_msgs::Marker::CUBE;
        break;
      case rc_tracking_msgs::Leg::LABEL_BKGD:
        m.type = visualization_msgs::Marker::CYLINDER;
        break;
    }
    rviz_markers.markers.push_back(m);

    // Comparison using '==' and not '>=' is important, as it allows
    // <max_detected_clusters_>=-1 to publish infinite markers
    if (clusters_published_counter == max_detected_clusters_)
      break;
  }

  // Clear remaining markers in Rviz
  for (int id_num_diff = num_prev_markers_published_ - id_num; id_num_diff > 0; id_num_diff--)
  {
    delete_m.header.stamp = scan->header.stamp;
    delete_m.header.frame_id = fixed_frame_;
    delete_m.ns = "laser_detected_clusters";
    delete_m.id = id_num_diff + id_num;
    delete_m.action = visualization_msgs::Marker::DELETE;
    rviz_markers.markers.push_back(delete_m);
  }
  num_prev_markers_published_ = id_num;  // For the next callback

  markers_pub_.publish(rviz_markers);
  detected_clusters_pub_.publish(detected_clusters);

  end = ros::WallTime::now();
  double exec_time = (end - start).toNSec() * 1e-6;
  execution_time_ += exec_time;
  avg_exec_time_ = execution_time_ / scan_num_;
  if (exec_time > max_exec_time_)
    max_exec_time_ = exec_time;
  ROS_INFO_STREAM("Scan" << scan_num_ << ": Execution time (ms): " << exec_time);
  ROS_INFO_STREAM("Max execution time (ms): " << max_exec_time_ << " Avg execution time (ms): " << avg_exec_time_);
}

bool CompareClusters::operator()(const rc_tracking_msgs::Leg& a, const rc_tracking_msgs::Leg& b)
{
  float rel_dist_a = pow(a.position.x * a.position.x + a.position.y * a.position.y, 1. / 2.);
  float rel_dist_b = pow(b.position.x * b.position.x + b.position.y * b.position.y, 1. / 2.);
  return rel_dist_a < rel_dist_b;
}

}  // namespace laser_human_tracker
}