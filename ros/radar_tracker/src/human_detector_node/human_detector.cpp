#include "human_detector.h"
#define PI 3.14159265359

namespace multi_posture_leg_tracker {
namespace radar_human_tracker
{
Sample* Sample::Extract(int ind, const sensor_msgs::PointCloud& scan)
{
  Sample* s = new Sample();

  s->angle = scan.channels[1].values[ind];
  s->intensity = scan.channels[0].values[ind];
  s->range = scan.channels[2].values[ind];
  s->x = scan.points[ind].x;
  s->y = scan.points[ind].y;
  return s;
}

void SampleSet::clear()
{
  for (SampleSet::iterator i = begin(); i != end(); ++i)
    delete (*i);
  std::set<Sample*, CompareSample>::clear();
}

void HumanDetector::createSample(const sensor_msgs::PointCloud& radar_scan)
{
  radar_human_tracker::SampleSet* cluster = new SampleSet;
  duplicated_samples_ = new SampleSet;

  for (int i = 0; i < radar_scan.channels[0].values.size(); i++)
  {
    Sample* s = Sample::Extract(i, radar_scan);

    if (s != NULL)
    {
      cluster->insert(s);
    }
  }

  int ind_sample = 0;
  int ind_angle = 0;
  float last_angle = 0;
  for (SampleSet::iterator it = cluster->begin(); it != cluster->end(); it++)
  {
    (*it)->index = ind_sample;

    // determine the angle index of points based on beams (used for clustering method)
    if (ind_sample > 0 && std::abs((*it)->angle - last_angle) > FLT_EPSILON)
      ind_angle++;
    (*it)->index_angle = ind_angle;
    last_angle = (*it)->angle;

    ind_sample++;

    Sample* s_duplicated = new Sample(*(*it));
    s_duplicated->angle -= 2 * M_PI;
    duplicated_samples_->insert(s_duplicated);
  }

  max_ind_angle_ = ind_angle + 1;
  for (SampleSet::iterator it = duplicated_samples_->begin(); it != duplicated_samples_->end(); it++)
  {
    (*it)->index_angle += max_ind_angle_;
  }

  clusters_.push_back(cluster);
}

void HumanDetector::radarCallback(const sensor_msgs::PointCloud::ConstPtr& radar_scan)
{
  scan_num_++;
  ros::WallTime start, end;
  start = ros::WallTime::now();

  createSample(*radar_scan);
  splitConnected(cluster_dist_euclid_);
  removeLessThan(min_points_per_cluster_);

  rc_tracking_msgs::LegArray detected_clusters;
  detected_clusters.header.frame_id = radar_scan->header.frame_id;
  detected_clusters.header.stamp = radar_scan->header.stamp;

  findTransformationTime(*radar_scan);

  visualization_msgs::MarkerArray rviz_markers;
  int id_num = 0;  // visualization markers id num

  // Store all processes clusters in a set ordered according to their relative
  // distance to the laser scanner
  std::set<rc_tracking_msgs::Leg, CompareClusters> cluster_set;
  if (!transform_available)
  {
    ROS_WARN("Not publishing detected clusters because no tf was available");
  }
  else
  {
    // Iterate through all clusters
    for (std::list<radar_human_tracker::SampleSet*>::iterator cluster = clusters_.begin(); cluster != clusters_.end();
         cluster++)
    {
      // Get position of cluster in radar frame
      tf::Stamped<tf::Point> position((*cluster)->getPosition(), tf_time, radar_scan->header.frame_id);
      float rel_dist = pow(position[0] * position[0] + position[1] * position[1], 1. / 2.);

      // Only consider clusters within max_distance.
      if (rel_dist < max_detect_distance_)
      {
        // Classify cluster
        std::vector<float> f = calcClusterFeatures(*cluster, *radar_scan, feature_set_size_);

        // feed the feature vector into classifier
        float label;
        float probability_of_human;
        classifier_->classifyFeatureVector(f, label, probability_of_human);
        // Consider only clusters that have a confidence greater than detection_threshold_
        if (probability_of_human > detection_threshold_)
        {
          // Transform cluster position to fixed frame
          // This should always be succesful because we've checked earlier if a
          // tf was available
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
            // Add detected cluster to set of detected clusters, along with
            // its relative position to the laser scanner
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
        std::list<Sample*> front_contour;
        std::list<Sample*> back_contour;
        std::list<Sample*> contour;
        extractContourOfCluster(*cluster, front_contour, back_contour, contour);
        visualization_msgs::Marker polygon_msg;
        polygon_msg.header.frame_id = fixed_frame_;
        polygon_msg.header.stamp = radar_scan->header.stamp;
        polygon_msg.ns = "radar_detected_clusters";
        polygon_msg.type = visualization_msgs::Marker::LINE_STRIP;
        polygon_msg.scale.x = 0.01;
        polygon_msg.color.r = 0.0;
        polygon_msg.color.g = 0.0;
        polygon_msg.color.b = 0.0;
        polygon_msg.color.a = 1.0;
        id_num++;
        polygon_msg.id = id_num;
        for (const auto& s : contour)
        {
          geometry_msgs::Point p;
          p.x = s->x;
          p.y = s->y;
          p.z = 0;
          polygon_msg.points.push_back(p);
        }
        geometry_msgs::Point first;
        first.x = contour.front()->x;
        first.y = contour.front()->y;
        first.z = 0;
        polygon_msg.points.push_back(first);  // connect first and last point for a closed contour

        rviz_markers.markers.push_back(polygon_msg);
      }
    }

    // Publish detected clusters to /radar_detected_clusters and to rviz
    // They are ordered from closest to the radar scanner to furthest
    int clusters_published_counter = 0;
    for (std::set<rc_tracking_msgs::Leg>::iterator it = cluster_set.begin(); it != cluster_set.end(); ++it)
    {
      // Publish to /radar_detected_clusters topic
      rc_tracking_msgs::Leg cluster = *it;
      detected_clusters.legs.push_back(cluster);
      clusters_published_counter++;

      // Publish marker to rviz
      id_num++;
      visualization_msgs::Marker m;
      m.header.stamp = radar_scan->header.stamp;
      m.header.frame_id = fixed_frame_;
      m.ns = "radar_detected_clusters";
      m.id = id_num;
      m.pose.position.x = cluster.position.x;
      m.pose.position.y = cluster.position.y;
      m.pose.position.z = 0.0;
      m.scale.x = 0.13;
      m.scale.y = 0.13;
      m.scale.z = 0.13;
      m.color.a = 1;
      m.color.r = 0;
      m.color.g = cluster.confidence;  // 0
      m.color.b = cluster.confidence;  // 1
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
      {
        break;
      }
    }

    // Clear remaining markers in Rviz
    for (int id_num_diff = num_prev_markers_published_ - id_num; id_num_diff > 0; id_num_diff--)
    {
      visualization_msgs::Marker m;
      m.header.stamp = radar_scan->header.stamp;
      m.header.frame_id = fixed_frame_;
      m.ns = "radar_detected_clusters";
      m.id = id_num_diff + id_num;
      m.action = m.DELETE;
      rviz_markers.markers.push_back(m);
    }
    num_prev_markers_published_ = id_num;  // For the next callback

    markers_pub_.publish(rviz_markers);
  }
  detected_clusters_pub_.publish(detected_clusters);

  clearAll();

  end = ros::WallTime::now();
  double exec_time = (end - start).toNSec() * 1e-6;
  execution_time_ += exec_time;
  avg_exec_time_ = execution_time_ / scan_num_;
  if (exec_time > max_exec_time_)
    max_exec_time_ = exec_time;
  ROS_INFO_STREAM("Scan" << scan_num_ << ": Execution time (ms): " << exec_time);
  ROS_INFO_STREAM("Max execution time (ms): " << max_exec_time_ << " Avg execution time (ms): " << avg_exec_time_);
  ROS_DEBUG_STREAM("Filtered scan size: " << radar_scan->points.size());
}

tf::Point SampleSet::getPosition()
{
  float x_mean = 0.0;
  float y_mean = 0.0;
  for (iterator i = begin(); i != end(); ++i)
  {
    x_mean += ((*i)->x) / size();
    y_mean += ((*i)->y) / size();
  }
  return tf::Point(x_mean, y_mean, 0.0);
}

HumanDetector::HumanDetector()
  : nh("~"), scan_num_(0), num_prev_markers_published_(0), execution_time_(0), avg_exec_time_(0), max_exec_time_(0)
{
  ROS_INFO("Radar Processor started");
  loadParameters();

  filtered_radar_data = nh.subscribe(scan_topic_, 1, &HumanDetector::radarCallback, this);
  markers_pub_ = nh.advertise<visualization_msgs::MarkerArray>(detected_clusters_marker_topic_, 5);
  detected_clusters_pub_ = nh.advertise<rc_tracking_msgs::LegArray>(detected_clusters_topic_, 5);
}

HumanDetector::HumanDetector(const sensor_msgs::PointCloud& scan)
{
  createSample(scan);
}

void HumanDetector::loadParameters()
{
  // Get ROS parameters
  if (!nh.getParam("model_file", model_file_))
    std::cout << "ERROR! Could not get model filename" << std::endl;
  if (!nh.getParam("scan_topic", scan_topic_))
    std::cout << "Parameter Error: scan_topic" << std::endl;
  if (!nh.getParam("detected_clusters_topic", detected_clusters_topic_))
    std::cout << "Parameter Error: detected_clusters_topic" << std::endl;
  if (!nh.getParam("detected_clusters_marker_topic", detected_clusters_marker_topic_))
    std::cout << "Parameter Error: detected_clusters_marker_topic" << std::endl;
  if (!nh.getParam("fixed_frame", fixed_frame_))
    std::cout << "Parameter Error: fixed_frame" << std::endl;
  if (!nh.getParam("detection_threshold", detection_threshold_))
    std::cout << "Parameter Error: detection_threshold_" << std::endl;
  if (!nh.getParam("cluster_dist_euclid", cluster_dist_euclid_))
    std::cout << "Parameter Error: cluster_dist_euclid" << std::endl;
  if (!nh.getParam("min_points_per_cluster", min_points_per_cluster_))
    std::cout << "Parameter Error: min_points_per_cluster" << std::endl;
  if (!nh.getParam("max_detect_distance", max_detect_distance_))
    std::cout << "Parameter Error: max_detect_distance" << std::endl;
  if (!nh.getParam("use_scan_header_stamp_for_tfs", use_scan_header_stamp_for_tfs_))
    std::cout << "Parameter Error: use_scan_header_stamp_for_tfs" << std::endl;
  if (!nh.getParam("max_detected_clusters", max_detected_clusters_))
    std::cout << "Parameter Error: max_detected_clusters_" << std::endl;
  if (!nh.getParam("classifier_type", classifier_type_))
    std::cout << "Parameter Error: classifier_type_" << std::endl;
  if (!nh.getParam("visualize_contour", visualize_contour_))
    std::cout << "Parameter Error: visualize_contour_" << std::endl;
  if (!nh.getParam("publish_background", publish_background_))
    std::cout << "Parameter Error: publish_background" << std::endl;

  if (classifier_type_ == "opencv_random_forest")
    classifier_ = std::make_shared<OpenCVRandomForestClassifier>(nh);
  else if (classifier_type_ == "opencv_adaboost")
    classifier_ = std::make_shared<OpenCVAdaBoostClassifier>(nh);
  else if (classifier_type_ == "mlpack_random_forest")
    classifier_ = std::make_shared<MlpackRandomForestClassifier>(nh);
  else if (classifier_type_ == "mlpack_adaboost")
    classifier_ = std::make_shared<MlpackAdaBoostClassifier>(nh);
  else
    ROS_ERROR("ERROR! The classifer type should be either opencv_random_forest or opencv_adaboost or "
              "mlpack_random_forest or mlpack_adaboost!");

  classifier_->loadModel(model_file_);
  feat_count_ = classifier_->getFeatureSetSize();  // returns the number of variables

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

  ROS_INFO("Parameteres loaded");
}

HumanDetector::~HumanDetector()
{
  std::list<SampleSet*>::iterator c_iter = clusters_.begin();
  while (c_iter != clusters_.end())
  {
    delete (*c_iter);
    clusters_.erase(c_iter++);
  }
}

void HumanDetector::turnClusters()
{
  std::list<SampleSet*> clusters_new_ = clusters_;
  int turns = 4;
  for (int n = 1; n != turns; n++)
  {
    int delta = round(360 / turns);

    for (auto iter = clusters_new_.begin(); iter != clusters_new_.end(); iter++)
    {
      radar_human_tracker::SampleSet* cluster_turn = new SampleSet;
      float x_mean = 0.0;
      float y_mean = 0.0;
      for (auto s_q = (*iter)->begin(); s_q != (*iter)->end(); s_q++)
      {
        x_mean += ((*s_q)->x) / (*iter)->size();
        y_mean += ((*s_q)->y) / (*iter)->size();
      }
      for (auto s_q = (*iter)->begin(); s_q != (*iter)->end(); s_q++)
      {
        Sample* s = new Sample;
        s->index = ((*s_q)->index);
        s->intensity = ((*s_q)->intensity);
        s->x = ((*s_q)->x - x_mean) * cos(n * delta * PI / 180) - ((*s_q)->y - y_mean) * sin(n * delta * PI / 180) +
               x_mean;
        s->y = ((*s_q)->x - x_mean) * sin(n * delta * PI / 180) + ((*s_q)->y - y_mean) * cos(n * delta * PI / 180) +
               y_mean;
        s->angle = atan2(s->y, s->x);
        s->range = sqrt(pow(s->x, 2) + pow(s->y, 2));

        cluster_turn->insert(s);
      }
      clusters_.push_back(cluster_turn);
    }
  }
}

void HumanDetector::removeLessThan(uint32_t num)
{
  std::list<SampleSet*>::iterator c_iter = clusters_.begin();
  while (c_iter != clusters_.end())
  {
    if ((*c_iter)->size() < num)
    {
      delete (*c_iter);
      clusters_.erase(c_iter++);
    }
    else
    {
      ++c_iter;
    }
  }
}

void HumanDetector::splitConnected(float thresh)
{
  // Holds our temporary list of split clusters because we will be modifying our
  // existing list in the mean time
  std::list<SampleSet*> tmp_clusters;
  std::list<SampleSet*>::iterator c_iter = clusters_.begin();

  std::list<std::set<int>> tmp_clusters_ind;

  while (c_iter != clusters_.end())
  {
    while ((*c_iter)->size() > 0)
    {
      // Iterate over radar scan samples in clusters_ and collect those which
      // are within a euclidian distance of <thresh> and store new clusters in
      // tmp_clusters
      bool is_rear_cluster = false;

      SampleSet::iterator s_first = (*c_iter)->begin();
      std::list<Sample*> sample_queue;
      sample_queue.push_back(*s_first);
      std::set<int> sample_queue_ind;
      sample_queue_ind.insert((*s_first)->index);
      (*c_iter)->erase(s_first);
      std::list<Sample*>::iterator s_q = sample_queue.begin();
      while (s_q != sample_queue.end())
      {
        float angle_increment = 0.025;
        int expand = static_cast<int>(std::floor(asin(thresh / (*s_q)->range) / angle_increment));

        SampleSet::iterator s_rest = (*c_iter)->begin();

        while (s_rest != (*c_iter)->end() and (*s_rest)->index_angle <= (*s_q)->index_angle + expand)
        {
          if ((*s_rest)->index_angle < (*s_q)->index_angle - expand)
          {
            ++s_rest;
            continue;
          }

          if ((*s_rest)->index_angle == (*s_q)->index_angle)
          {
            if (std::abs((*s_rest)->range - (*s_q)->range) < thresh)
            {
              sample_queue.push_back(*s_rest);
              sample_queue_ind.insert((*s_rest)->index);
              (*c_iter)->erase(s_rest++);
            }
            else
            {
              ++s_rest;
            }
          }
          else if (sqrt(pow((*s_q)->x - (*s_rest)->x, 2.0f) + pow((*s_q)->y - (*s_rest)->y, 2.0f)) < thresh)
          {
            sample_queue.push_back(*s_rest);
            sample_queue_ind.insert((*s_rest)->index);
            (*c_iter)->erase(s_rest++);
          }
          else
          {
            ++s_rest;
          }
        }

        if (s_rest == (*c_iter)->end())
        {
          is_rear_cluster = true;

          SampleSet::iterator s_rest_dup = duplicated_samples_->begin();

          while (s_rest_dup != duplicated_samples_->end() and
                 (*s_rest_dup)->index_angle <= (*s_q)->index_angle + expand)
          {
            if (sqrt(pow((*s_q)->x - (*s_rest_dup)->x, 2.0f) + pow((*s_q)->y - (*s_rest_dup)->y, 2.0f)) < thresh)
            {
              sample_queue.push_back(*s_rest_dup);
              sample_queue_ind.insert((*s_rest_dup)->index);
              duplicated_samples_->erase(s_rest_dup++);
            }
            else
            {
              ++s_rest_dup;
            }
          }
        }

        s_q++;
      }

      // Move all the samples into the new cluster
      SampleSet* c = new SampleSet;
      for (s_q = sample_queue.begin(); s_q != sample_queue.end(); s_q++)
      {
        c->insert(*s_q);
      }

      if (is_rear_cluster)
      {
        std::list<std::set<int>>::iterator it_ind = tmp_clusters_ind.begin();
        std::list<SampleSet*>::iterator it_cluster = tmp_clusters.begin();

        while (it_ind != tmp_clusters_ind.end())
        {
          if (std::includes(sample_queue_ind.begin(), sample_queue_ind.end(), (*it_ind).begin(), (*it_ind).end()))
          {
            tmp_clusters_ind.erase(it_ind);
            tmp_clusters.erase(it_cluster);
            break;
          }
          it_ind++;
          it_cluster++;
        }
      }

      // Store the temporary clusters
      tmp_clusters.push_back(c);
      tmp_clusters_ind.push_back(sample_queue_ind);
    }

    // Now that c_iter is empty, we can delete
    delete (*c_iter);
    c_iter++;
  }

  clusters_.clear();
  // Insert our temporary clusters list back into the de facto list
  clusters_.insert(clusters_.begin(), tmp_clusters.begin(), tmp_clusters.end());
}

bool CompareClusters::operator()(const rc_tracking_msgs::Leg& a, const rc_tracking_msgs::Leg& b)
{
  float rel_dist_a = pow(a.position.x * a.position.x + a.position.y * a.position.y, 1. / 2.);
  float rel_dist_b = pow(b.position.x * b.position.x + b.position.y * b.position.y, 1. / 2.);
  return rel_dist_a < rel_dist_b;
}

void HumanDetector::clearAll()
{
  std::list<SampleSet*>::iterator c_iter = clusters_.begin();
  while (c_iter != clusters_.end())
  {
    (*c_iter)->clear();
    delete (*c_iter);
    clusters_.erase(c_iter++);
  }
  duplicated_samples_->clear();
  delete duplicated_samples_;
}

void HumanDetector::findTransformationTime(const sensor_msgs::PointCloud& radar_scan)
{
  // Find out the time that should be used for tfs
  // Use time from scan header

  if (use_scan_header_stamp_for_tfs_)
  {
    tf_time = radar_scan.header.stamp;

    try
    {
      tfl_.waitForTransform(fixed_frame_, radar_scan.header.frame_id, tf_time, ros::Duration(5.0));
      transform_available = tfl_.canTransform(fixed_frame_, radar_scan.header.frame_id, tf_time);
    }
    catch (tf::TransformException ex)
    {
      ROS_WARN("Radar Detector: No tf available");
      transform_available = false;
    }
  }
  else
  {
    // Otherwise just use the latest tf available
    tf_time = ros::Time(0);
    transform_available = tfl_.canTransform(fixed_frame_, radar_scan.header.frame_id, tf_time);
  }
}

void HumanDetector::extractContourOfCluster(const SampleSet* cluster, std::list<Sample*>& front_contour,
                                            std::list<Sample*>& back_contour, std::list<Sample*>& contour)
{
  // sort radar points in the cluster by beams
  std::map<int, radar_human_tracker::SampleSet*> map{};
  int last_index_angle = -1;
  for (radar_human_tracker::SampleSet::iterator i = cluster->begin(); i != cluster->end(); i++)
  {
    int ind = (*i)->index_angle;
    if (ind != last_index_angle)
    {
      SampleSet* s = new SampleSet;
      s->insert(*i);
      map.emplace(ind, s);
      last_index_angle = ind;
    }
    else
    {
      map[ind]->insert(*i);
    }
  }

  std::map<int, radar_human_tracker::SampleSet*>::iterator it_map = map.begin();

  // If there is only one point on the beam, it belongs to the front contour
  while (it_map != map.end())
  {
    SampleSet* cur_beam = it_map->second;
    if (cur_beam->size() == 1)
    {
      front_contour.emplace_back(*(cur_beam->begin()));
    }
    else
    {
      front_contour.emplace_back(*(cur_beam->begin()));   // first point on the current beam
      back_contour.emplace_front(*(cur_beam->rbegin()));  // last point on the current beam
    }
    it_map++;
  }

  // combine front and back contour as the cluster contour
  contour.insert(contour.end(), front_contour.begin(), front_contour.end());
  contour.insert(contour.end(), back_contour.begin(), back_contour.end());
}

std::vector<float> HumanDetector::calcClusterFeatures(const radar_human_tracker::SampleSet* cluster,
                                                      const sensor_msgs::PointCloud& scan, int feature_set_size)
{
  // Points on the contour
  std::list<Sample*> front_contour;
  std::list<Sample*> back_contour;
  std::list<Sample*> contour;
  extractContourOfCluster(cluster, front_contour, back_contour, contour);

  /**
   * Feature: Number of points on contour
   */
  int num_points_contour = contour.size();

  /**
   * Feature: Number of points
   */
  int num_points = cluster->size();

  // Compute mean and median points for future use
  float x_mean = 0.0;
  float y_mean = 0.0;
  std::vector<float> x_median_set;
  std::vector<float> y_median_set;
  for (radar_human_tracker::SampleSet::iterator i = cluster->begin(); i != cluster->end(); i++)
  {
    x_mean += ((*i)->x) / num_points;
    y_mean += ((*i)->y) / num_points;
    x_median_set.push_back((*i)->x);
    y_median_set.push_back((*i)->y);
  }

  sort(x_median_set.begin(), x_median_set.end());
  sort(y_median_set.begin(), y_median_set.end());

  float x_median = 0.5 * (*(x_median_set.begin() + (num_points - 1) / 2) + *(x_median_set.begin() + num_points / 2));
  float y_median = 0.5 * (*(y_median_set.begin() + (num_points - 1) / 2) + *(y_median_set.begin() + num_points / 2));

  /**
   * Feature: std from mean point
   * Feature: mean average deviation from median point
   */
  double sum_std_diff = 0.0;
  double sum_med_diff = 0.0;
  double std_x = 0.0;
  double std_y = 0.0;
  // used for computing kurtosis
  double sum_kurt = 0.0;

  for (radar_human_tracker::SampleSet::iterator i = cluster->begin(); i != cluster->end(); i++)
  {
    double diff_x = pow((*i)->x - x_mean, 2);
    double diff_y = pow((*i)->y - y_mean, 2);
    std_x += diff_x;
    std_y += diff_y;
    sum_kurt += pow(diff_x + diff_y, 2);  

    sum_med_diff += sqrt(pow((*i)->x - x_median, 2) + pow((*i)->y - y_median, 2));
  }
  sum_std_diff = std_x + std_y;
  float std = sqrt(1.0 / (num_points - 1.0) * sum_std_diff);
  float avg_median_dev = sum_med_diff / num_points;

  // used for computing aspect ratio
  std_x = sqrt(std_x / (num_points - 1.0));
  std_y = sqrt(std_y / (num_points - 1.0));

  // Get first and last points in cluster
  radar_human_tracker::SampleSet::iterator first = cluster->begin();
  radar_human_tracker::SampleSet::iterator last = cluster->end();
  --last;

  /**
   * Feature 04: width
   */
  float width = sqrt(pow((*first)->x - (*last)->x, 2) + pow((*first)->y - (*last)->y, 2));

  /**
   * Feature: linearity
   * Feature: min_lin_err: min error of points to fitted line
   * Feature: max_lin_err: max error of points to fitted line
   * Feature: ratio_min_max_lin_err
   */
  CvMat* points = cvCreateMat(num_points, 2, CV_64FC1);
  {
    int j = 0;
    for (radar_human_tracker::SampleSet::iterator i = cluster->begin(); i != cluster->end(); i++)
    {
      cvmSet(points, j, 0, (*i)->x - x_mean);
      cvmSet(points, j, 1, (*i)->y - y_mean);
      j++;
    }
  }

  CvMat* W = cvCreateMat(2, 2, CV_64FC1);
  CvMat* U = cvCreateMat(num_points, 2, CV_64FC1);
  CvMat* V = cvCreateMat(2, 2, CV_64FC1);
  cvSVD(points, W, U, V);

  CvMat* rot_points = cvCreateMat(num_points, 2, CV_64FC1);
  cvMatMul(U, W, rot_points);

  float linearity = 0.0;
  float min_lin_err = std::numeric_limits<float>::infinity();
  float max_lin_err = 0.0;
  float ratio_min_max_lin_err = 0.0;
  for (int i = 0; i < num_points; i++)
  {
    float diff = cvmGet(rot_points, i, 1);
    if (diff < min_lin_err)
      min_lin_err = diff;
    if (diff > max_lin_err)
      max_lin_err = diff;
    linearity += pow(diff, 2);
  }
  if (max_lin_err != 0.0)
    ratio_min_max_lin_err = min_lin_err / max_lin_err;

  cvReleaseMat(&points);
  points = 0;
  cvReleaseMat(&W);
  W = 0;
  cvReleaseMat(&U);
  U = 0;
  cvReleaseMat(&V);
  V = 0;
  cvReleaseMat(&rot_points);
  rot_points = 0;

  /**
   * Feature: circularity
   * Feature: min_dist2center: min difference from points to center of fitted circle
   * Feature: max_dist2center: max difference from points to center of fitted circle
   * Feature: ratio_min_max_dist2center
   */
  CvMat* A = cvCreateMat(num_points, 3, CV_64FC1);
  CvMat* B = cvCreateMat(num_points, 1, CV_64FC1);
  {
    int j = 0;
    for (radar_human_tracker::SampleSet::iterator i = cluster->begin(); i != cluster->end(); i++)
    {
      float x = (*i)->x;
      float y = (*i)->y;

      cvmSet(A, j, 0, -2.0 * x);
      cvmSet(A, j, 1, -2.0 * y);
      cvmSet(A, j, 2, 1);

      cvmSet(B, j, 0, -pow(x, 2) - pow(y, 2));
      j++;
    }
  }
  CvMat* sol = cvCreateMat(3, 1, CV_64FC1);

  cvSolve(A, B, sol, CV_SVD);

  float xc = cvmGet(sol, 0, 0);
  float yc = cvmGet(sol, 1, 0);
  float rc = sqrt(pow(xc, 2) + pow(yc, 2) - cvmGet(sol, 2, 0));

  cvReleaseMat(&A);
  A = 0;
  cvReleaseMat(&B);
  B = 0;
  cvReleaseMat(&sol);
  sol = 0;

  float circularity = 0.0;
  float min_dist2center = std::numeric_limits<float>::infinity();
  float max_dist2center = 0.0;
  float ratio_min_max_dist2center = 0.0;

  for (radar_human_tracker::SampleSet::iterator i = cluster->begin(); i != cluster->end(); i++)
  {
    float dist = sqrt(pow(xc - (*i)->x, 2) + pow(yc - (*i)->y, 2));
    if (dist < min_dist2center)
      min_dist2center = dist;
    if (dist > max_dist2center)
      max_dist2center = dist;
    circularity += pow(rc - dist, 2);
  }

  if (max_dist2center != 0.0)
    ratio_min_max_dist2center = min_dist2center / max_dist2center;

  /**
   * Feature: radius
   */
  float radius = rc;

  /**
   * Feature: boundary length (1D)
   * Feature: boundary regularity (1D)s
   */
  float boundary_length = 0.0;
  float last_boundary_seg = 0.0;

  float boundary_regularity = 0.0;
  double sum_boundary_reg_sq = 0.0;

  /**
   * Feature: mean curvature (1D)
   */
  float mean_curvature = 0.0;

  /**
   * Feature: mean angular difference (1D)
   */
  float ang_diff = 0.0;

  if (num_points_contour >= 3)
  {
    std::list<Sample*>::iterator left_ctr = contour.begin();
    left_ctr++;
    left_ctr++;
    std::list<Sample*>::iterator mid_ctr = contour.begin();
    mid_ctr++;
    std::list<Sample*>::iterator right_ctr = contour.begin();

    while (left_ctr != contour.end())
    {
      float mlx = (*left_ctr)->x - (*mid_ctr)->x;
      float mly = (*left_ctr)->y - (*mid_ctr)->y;
      float L_ml = sqrt(mlx * mlx + mly * mly);

      float mrx = (*right_ctr)->x - (*mid_ctr)->x;
      float mry = (*right_ctr)->y - (*mid_ctr)->y;
      float L_mr = sqrt(mrx * mrx + mry * mry);

      float lrx = (*left_ctr)->x - (*right_ctr)->x;
      float lry = (*left_ctr)->y - (*right_ctr)->y;
      float L_lr = sqrt(lrx * lrx + lry * lry);

      boundary_length += L_mr;
      sum_boundary_reg_sq += L_mr * L_mr;
      last_boundary_seg = L_ml;

      float A = (mlx * mrx + mly * mry) / pow(L_mr, 2);
      float B = (mlx * mry - mly * mrx) / pow(L_mr, 2);

      float th = atan2(B, A);

      if (th < 0)
        th += 2 * M_PI;

      ang_diff += th / num_points_contour;

      float s = 0.5 * (L_ml + L_mr + L_lr);
      float area_square = std::max(s * (s - L_ml) * (s - L_mr) * (s - L_lr), static_cast<float>(0));
      float area = sqrt(area_square);

      if (th > 0)
        mean_curvature += 4 * (area) / (L_ml * L_mr * L_lr * num_points_contour);
      else
        mean_curvature -= 4 * (area) / (L_ml * L_mr * L_lr * num_points_contour);

      left_ctr++;
      mid_ctr++;
      right_ctr++;
    }

    boundary_length += last_boundary_seg;
    sum_boundary_reg_sq += last_boundary_seg * last_boundary_seg;

    boundary_regularity =
        sqrt((sum_boundary_reg_sq - pow(boundary_length, 2) / num_points_contour) / (num_points_contour - 1));
  }
  /**
   * Feature: incribed angle variance (1D)
   * Feature: std incribed angle variance (1D)
   */
  float iav = 0.0;
  float std_iav = 0.0;

  double sum_iav = 0.0;
  double sum_iav_sq = 0.0;

  if (front_contour.size() >= 3 && back_contour.size() >= 3)
  {
    std::list<Sample*>::iterator first_ctr = front_contour.begin();  // first point on contour
    std::list<Sample*>::iterator mid_ctr = front_contour.begin();
    mid_ctr++;
    std::list<Sample*>::iterator last_ctr = front_contour.end();
    last_ctr--;

    while (mid_ctr != last_ctr)
    {
      float mlx = (*first_ctr)->x - (*mid_ctr)->x;
      float mly = (*first_ctr)->y - (*mid_ctr)->y;

      float mrx = (*last_ctr)->x - (*mid_ctr)->x;
      float mry = (*last_ctr)->y - (*mid_ctr)->y;
      float L_mr = sqrt(mrx * mrx + mry * mry);

      float A = (mlx * mrx + mly * mry) / pow(L_mr, 2);
      float B = (mlx * mry - mly * mrx) / pow(L_mr, 2);

      float th = atan2(B, A);

      if (th < 0)
      {
        th += 2 * M_PI;
      }
      sum_iav += th;
      sum_iav_sq += th * th;

      mid_ctr++;
    }

    first_ctr = back_contour.begin();  // first point on contour
    mid_ctr = back_contour.begin();
    mid_ctr++;
    last_ctr = back_contour.end();
    last_ctr--;
    while (mid_ctr != last_ctr)
    {
      float mlx = (*first_ctr)->x - (*mid_ctr)->x;
      float mly = (*first_ctr)->y - (*mid_ctr)->y;

      float mrx = (*last_ctr)->x - (*mid_ctr)->x;
      float mry = (*last_ctr)->y - (*mid_ctr)->y;
      float L_mr = sqrt(mrx * mrx + mry * mry);

      float A = (mlx * mrx + mly * mry) / pow(L_mr, 2);
      float B = (mlx * mry - mly * mrx) / pow(L_mr, 2);

      float th = atan2(B, A);

      if (th < 0)
      {
        th += 2 * M_PI;
      }
      sum_iav += th;
      sum_iav_sq += th * th;

      mid_ctr++;
    }

    iav = sum_iav / num_points_contour;
    std_iav = sqrt((sum_iav_sq - pow(sum_iav, 2) / num_points_contour) / (num_points_contour - 1));
  }
  /**
   * Feature: aspect ratio (1D)
   */
  float aspect_ratio = (1.0 + MIN(std_x, std_y)) / (1.0 + MAX(std_x, std_y));

  /**
   * Feature: kurtosis (1D)
   */
  float num = sum_kurt / num_points;
  float std_n = sqrt(1.0 / num_points * sum_std_diff);
  float denom = pow(std_n, 4);
  float kurtosis = num / denom;

  /**
   * Feature: polygon area (1D)
   */
  double sum_area = 0.0;
  std::list<Sample*>::iterator right_ctr = contour.begin();
  std::list<Sample*>::iterator mid_ctr = contour.begin();
  mid_ctr++;
  while (mid_ctr != contour.end())
  {
    sum_area += ((*right_ctr)->x * (*mid_ctr)->y) - ((*mid_ctr)->x * (*right_ctr)->y);
    right_ctr++;
    mid_ctr++;
  }
  // close polygon
  std::list<Sample*>::iterator first_ctr = contour.begin();
  std::list<Sample*>::iterator last_ctr = contour.end();
  last_ctr--;

  sum_area += ((*last_ctr)->x * (*first_ctr)->y) - ((*first_ctr)->x * (*last_ctr)->y);
  sum_area *= 0.5;
  float polygon_area = std::abs(sum_area);

  /**
   * Feature: distance to sensor
   */
  float distance = sqrt(x_median * x_median + y_median * y_median);

  // mean and median intensity
  float int_mean = 0;
  std::vector<float> int_median_set;
  for (auto i = cluster->begin(); i != cluster->end(); i++)
  {
    int_mean += ((*i)->intensity);
    int_median_set.push_back((*i)->intensity);
  }
  int_mean = int_mean / num_points;

  sort(int_median_set.begin(), int_median_set.end());

  float int_median =
      0.5 * (*(int_median_set.begin() + (num_points - 1) / 2) + *(int_median_set.begin() + num_points / 2));

  // Compute std_intensity
  double sum_std_diff_int = 0.0;

  for (auto i = cluster->begin(); i != cluster->end(); i++)
  {
    sum_std_diff_int += pow((*i)->intensity - int_mean, 2);
  }

  float std_int = sqrt(1.0 / (num_points - 1.0) * sum_std_diff_int);

  // Compute max and min intensity
  float max_int = -1000;
  float min_int = 0;

  for (auto i = cluster->begin(); i != cluster->end(); i++)
  {
    if ((*i)->intensity < min_int)
    {
      min_int = (*i)->intensity;
    }

    if ((*i)->intensity > max_int)
    {
      max_int = (*i)->intensity;
    }
  }

  // Compute alpha shapes
  std::list<Point> alpha_points;
  std::vector<Point> alpha_vector;
  std::vector<double> int_vec;
  for (auto i = cluster->begin(); i != cluster->end(); i++)
  {
    Point point((*i)->x, (*i)->y);
    alpha_points.push_back(point);
    alpha_vector.push_back(point);
    int_vec.push_back((*i)->intensity);
  }
  Alpha_shape_2* Z;
  Z = new Alpha_shape_2(alpha_points.begin(), alpha_points.end(), FT(10000), Alpha_shape_2::GENERAL);
  std::vector<Segment> segments;
  alpha_edges(*Z, std::back_inserter(segments));
  auto f = *(alpha_points).begin();

  std::vector<double> int_bound;
  for (auto i = segments.begin(); i != segments.end(); i++)
  {
    auto d = i->target();
    double x_value = d.x();
    double y_value = d.y();
    Point point2(x_value, y_value);
    for (int i = 0; i != alpha_vector.size(); i++)
    {
      if (alpha_vector[i].x() == x_value && alpha_vector[i].y() == y_value)
      {
        int_bound.push_back(int_vec[i]);
      }
    }
  }
  float int_bound_mean;
  int_bound_mean = std::accumulate(std::begin(int_bound), std::end(int_bound), 0.0) / int_bound.size();

  // Add features
  std::vector<float> features;

  switch (feature_set_size)
  {
    case 0:
      // features from "Using Boosted Features for the Detection of People in 2D Range Data"
      features.push_back(num_points);          // 1
      features.push_back(num_points_contour);  // 2
      features.push_back(std);                 // 3
      features.push_back(avg_median_dev);      // 4

      features.push_back(width);  // 5

      features.push_back(linearity);              // 6
      features.push_back(min_lin_err);            // 7
      features.push_back(max_lin_err);            // 8
      features.push_back(ratio_min_max_lin_err);  // 9

      features.push_back(circularity);                // 10
      features.push_back(radius);                     // 11
      features.push_back(min_dist2center);            // 12
      features.push_back(max_dist2center);            // 13
      features.push_back(ratio_min_max_dist2center);  // 14

      features.push_back(boundary_length);      // 15
      features.push_back(boundary_regularity);  // 16

      features.push_back(mean_curvature);  // 17

      features.push_back(ang_diff);  // 18

      features.push_back(iav);      // 19
      features.push_back(std_iav);  // 20

      // new features from L. Spinello
      features.push_back(aspect_ratio);  // 21

      features.push_back(kurtosis);  // 22

      features.push_back(polygon_area);  // 23

      // Intensity features
      features.push_back(int_mean);        // 24
      features.push_back(int_median);      // 25
      features.push_back(std_int);         // 26
      features.push_back(min_int);         // 27
      features.push_back(max_int);         // 28
      features.push_back(int_bound_mean);  // 29
      break;

    case 1:
      features.push_back(num_points);                 // 1
      features.push_back(num_points_contour);         // 2
      features.push_back(std);                        // 3
      features.push_back(avg_median_dev);             // 4
      features.push_back(width);                      // 5
      features.push_back(linearity);                  // 6
      features.push_back(min_lin_err);                // 7
      features.push_back(max_lin_err);                // 8
      features.push_back(ratio_min_max_lin_err);      // 9
      features.push_back(circularity);                // 10
      features.push_back(radius);                     // 11
      features.push_back(min_dist2center);            // 12
      features.push_back(max_dist2center);            // 13
      features.push_back(ratio_min_max_dist2center);  // 14
      features.push_back(boundary_length);            // 15
      features.push_back(boundary_regularity);        // 16
      features.push_back(mean_curvature);             // 17
      features.push_back(ang_diff);                   // 18
      features.push_back(iav);                        // 19
      features.push_back(std_iav);                    // 20
      features.push_back(aspect_ratio);               // 21
      features.push_back(kurtosis);                   // 22
      features.push_back(polygon_area);               // 23
      features.push_back(int_mean);                   // 24
      features.push_back(int_median);                 // 25
      features.push_back(std_int);                    // 26
      features.push_back(min_int);                    // 27
      features.push_back(max_int);                    // 28
      features.push_back(int_bound_mean);             // 29
      features.push_back(distance);                   // 30
      break;

    case 2:
      features.push_back(num_points);                 // 1*
      features.push_back(num_points_contour);         // 2*
      features.push_back(std);                        // 3
      features.push_back(avg_median_dev);             // 4
      features.push_back(width);                      // 5
      features.push_back(linearity);                  // 6
      features.push_back(min_lin_err);                // 7
      features.push_back(max_lin_err);                // 8
      features.push_back(ratio_min_max_lin_err);      // 9
      features.push_back(circularity);                // 10
      features.push_back(radius);                     // 11
      features.push_back(min_dist2center);            // 12
      features.push_back(max_dist2center);            // 13
      features.push_back(ratio_min_max_dist2center);  // 14
      features.push_back(boundary_length);            // 15
      features.push_back(boundary_regularity);        // 16
      features.push_back(mean_curvature);             // 17*
      features.push_back(ang_diff);                   // 18*
      features.push_back(iav);                        // 19*
      features.push_back(std_iav);                    // 20
      features.push_back(aspect_ratio);               // 21
      features.push_back(kurtosis);                   // 22
      features.push_back(polygon_area);               // 23
      features.push_back(int_mean);                   // 24*
      features.push_back(int_median);                 // 25*
      features.push_back(std_int);                    // 26*
      features.push_back(min_int);                    // 27
      features.push_back(max_int);                    // 28*
      features.push_back(int_bound_mean);             // 29

      features.push_back(num_points * distance);                 // 1*
      features.push_back(num_points_contour * distance);         // 2*
      features.push_back(ratio_min_max_dist2center / distance);  // 14*
      features.push_back(boundary_regularity / distance);        // 16*
      features.push_back(mean_curvature * distance);             // 17*
      features.push_back(ang_diff * distance);                   // 18*
      features.push_back(iav * distance);                        // 19*
      features.push_back(int_mean * distance);                   // 24*
      features.push_back(int_median * distance);                 // 25*
      features.push_back(std_int * distance);                    // 26*
      features.push_back(max_int * distance);                    // 28*

      features.push_back(linearity / num_points);                  // 6
      features.push_back(min_lin_err * num_points);                // 7
      features.push_back(max_lin_err / num_points);                // 8
      features.push_back(circularity / num_points);                // 10
      features.push_back(ratio_min_max_dist2center * num_points);  // 14
      features.push_back(boundary_length / num_points);            // 15
      features.push_back(mean_curvature / num_points);             // 17*
      features.push_back(ang_diff / num_points);                   // 18*
      features.push_back(iav / num_points);                        // 19*
      features.push_back(polygon_area / num_points);               // 23
      features.push_back(int_mean / num_points);                   // 24*
      features.push_back(int_median / num_points);                 // 25*
      features.push_back(std_int / num_points);                    // 26*
      features.push_back(max_int / num_points);                    // 28*

      break;

    case 3:
      features.push_back(num_points);                 // 1
      features.push_back(num_points_contour);         // 2
      features.push_back(std);                        // 3
      features.push_back(avg_median_dev);             // 4
      features.push_back(width);                      // 5
      features.push_back(linearity);                  // 6
      features.push_back(min_lin_err);                // 7
      features.push_back(max_lin_err);                // 8
      features.push_back(ratio_min_max_lin_err);      // 9
      features.push_back(circularity);                // 10
      features.push_back(radius);                     // 11
      features.push_back(min_dist2center);            // 12
      features.push_back(max_dist2center);            // 13
      features.push_back(ratio_min_max_dist2center);  // 14
      features.push_back(boundary_length);            // 15
      features.push_back(boundary_regularity);        // 16
      features.push_back(mean_curvature);             // 17
      features.push_back(ang_diff);                   // 18
      features.push_back(iav);                        // 19
      features.push_back(std_iav);                    // 20
      features.push_back(aspect_ratio);               // 21
      features.push_back(kurtosis);                   // 22
      features.push_back(polygon_area);               // 23
      features.push_back(int_mean);                   // 24
      features.push_back(int_median);                 // 25
      features.push_back(std_int);                    // 26
      features.push_back(min_int);                    // 27
      features.push_back(max_int);                    // 28
      features.push_back(int_bound_mean);             // 29

      for (uint i = 0; i < FEATURE_SET_0; i++)
      {
        features.push_back(features[i] / distance);
      }
      for (uint i = 0; i < FEATURE_SET_0; i++)
      {
        features.push_back(features[i] * distance);
      }
      for (uint i = 1; i < FEATURE_SET_0; i++)
      {
        features.push_back(features[i] / num_points);
      }
      for (uint i = 1; i < FEATURE_SET_0; i++)
      {
        features.push_back(features[i] * num_points);
      }
      break;

    case 4:
      features.push_back(num_points);
      features.push_back(std);
      features.push_back(avg_median_dev);
      features.push_back(width);
      features.push_back(linearity);
      features.push_back(circularity);
      features.push_back(radius);
      features.push_back(boundary_length);
      features.push_back(boundary_regularity);
      features.push_back(mean_curvature);
      features.push_back(ang_diff);
      features.push_back(iav);
      features.push_back(std_iav);
      features.push_back(distance);
      features.push_back(distance / num_points);
      features.push_back(int_mean);
      features.push_back(int_median);
      features.push_back(std_int);
      features.push_back(min_int);
      features.push_back(max_int);
      features.push_back(int_bound_mean);
      break;
  }

  return features;
}

template <class OutputIterator>
void alpha_edges(const Alpha_shape_2& A, OutputIterator out)
{
  Alpha_shape_edges_iterator it = A.alpha_shape_edges_begin(), end = A.alpha_shape_edges_end();
  for (; it != end; ++it)
    *out++ = A.segment(*it);
}
} 
}  // namespace radar_human_tracker
