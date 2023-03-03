#include "extract_feature_matrix.h"

namespace multi_posture_leg_tracker
{

namespace radar_human_tracker
{

ExtractFeatureMatrix::ExtractFeatureMatrix(ros::NodeHandle nh)
{
  // Get ROS params (all have default values so it's not critical we get them
  // all)
  nh.param("range_min", range_min_, 0.5);
  nh.param("range_max", range_max_, 8.0);
  nh.param("cluster_dist_euclid", cluster_dist_euclid_, 0.45);
  nh.param("min_points_per_cluster", min_points_per_cluster_, 5);
  nh.param("undersample_negative_factor", undersample_negative_factor_, 50);
  nh.param("feature_set_size", feature_set_size_, 0);

  // Print back params:
  printf("\nROS parameters: \n");
  printf("feature_set_size: %d \n", feature_set_size_);
  printf("cluster_dist_euclid:%.2fm \n", cluster_dist_euclid_);
  printf("min_points_per_cluster:%i \n", min_points_per_cluster_);
  printf("undersample_negative_factor:%i \n", undersample_negative_factor_);
  printf("\n");
}

void ExtractFeatureMatrix::run()
{
  saveCvMatrix(data_);
}

void ExtractFeatureMatrix::loadData(int argc, char** argv)
{
  // Parse command line arguements and load data
  printf("\nLoading data...\n");
  for (int i = 0; i < argc; i++)
  {
    if (!strcmp(argv[i], "--leg"))
    {
      char* rosbag_file = argv[++i];
      char* scan_topic = argv[++i];
      char* cluster_topic = argv[++i];
      loadAnnotatedData(rosbag_file, scan_topic, cluster_topic, data_);
    }
    else if (!strcmp(argv[i], "--squat"))
    {
      char* rosbag_file = argv[++i];
      char* scan_topic = argv[++i];
      char* cluster_topic = argv[++i];
      loadAnnotatedData(rosbag_file, scan_topic, cluster_topic, data_);
    }
    else if (!strcmp(argv[i], "--neg"))
    {
      char* rosbag_file = argv[++i];
      char* scan_topic = argv[++i];
      loadNegData(rosbag_file, scan_topic, data_);
    }
    else if (!strcmp(argv[i], "--neg_annotated"))
    {
      char* rosbag_file = argv[++i];
      char* scan_topic = argv[++i];
      char* cluster_topic = argv[++i];
      loadAnnotatedData(rosbag_file, scan_topic, cluster_topic, data_);
    }
    else if (!strcmp(argv[i], "--save_file_path"))
    {
      save_file_path_ = std::string(argv[++i]);
    }
  }

  // Check we have a valid save file
  if (save_file_path_.empty())
  {
    ROS_ERROR("Save file not specified properly in command line arguments \nExiting");
    exit(1);
  }

  printf("\n  Total samples: %i \n", (int)data_.size());
}

void ExtractFeatureMatrix::loadAnnotatedData(const char* rosbag_file, const char* scan_topic, const char* cluster_topic,
                                             std::vector<std::vector<float>>& data)
{
  int message_num = 0;
  int initial_pos_data_size = (int)data.size();

  rosbag::Bag bag;
  bag.open(rosbag_file, rosbag::bagmode::Read);
  std::vector<sensor_msgs::PointCloud::ConstPtr> scans;
  std::vector<geometry_msgs::PoseArray::ConstPtr> ground_truth_cluster_poses;

  rosbag::View view_1(bag, rosbag::TopicQuery(std::string(scan_topic)));
  BOOST_FOREACH (rosbag::MessageInstance const m, view_1)
  {
    sensor_msgs::PointCloud::ConstPtr scan = m.instantiate<sensor_msgs::PointCloud>();
    scans.push_back(scan);
  }

  rosbag::View view_2(bag, rosbag::TopicQuery(std::string(cluster_topic)));
  BOOST_FOREACH (rosbag::MessageInstance const m, view_2)
  {
    geometry_msgs::PoseArray::ConstPtr pose_array_msg = m.instantiate<geometry_msgs::PoseArray>();
    ground_truth_cluster_poses.push_back(pose_array_msg);
  }

  std::vector<sensor_msgs::PointCloud::ConstPtr>::iterator it_scans = scans.begin();
  std::vector<geometry_msgs::PoseArray::ConstPtr>::iterator it_poses = ground_truth_cluster_poses.begin();
  while (it_scans != scans.end())
  {
    sensor_msgs::PointCloud::ConstPtr scan = *it_scans;
    geometry_msgs::PoseArray::ConstPtr gt_cluster_pose = *it_poses;

    if (scan->points.size() and gt_cluster_pose->poses.size())
    {
      sensor_msgs::PointCloud filtered_positive_scan;
      filtered_positive_scan.header = scan->header;
      filtered_positive_scan.channels.resize(scan->channels.size());

      sensor_msgs::PointCloud filtered_positive_scan_distance;
      filtered_positive_scan_distance.header = scan->header;
      filtered_positive_scan_distance.channels.resize(scan->channels.size());

      for (int i = 0; i != scan->channels.size(); i++)
      {
        filtered_positive_scan.channels[i].name = scan->channels[i].name;
        filtered_positive_scan_distance.channels[i].name = scan->channels[i].name;
      }

      otsu_filter::OtsuFilter filter;
      filter.filterRadar(*scan, filtered_positive_scan_distance, range_min_, range_max_, -100 * PI / 180,
                         -260 * PI / 180);
      if (filtered_positive_scan_distance.points.size() == 0)
        break;
      int min_intensity = filter.getOtsuThreshold(&filtered_positive_scan_distance, 0);
      filter.filterRadarIntensity(filtered_positive_scan_distance, filtered_positive_scan, min_intensity - 20);

      HumanDetector processor(filtered_positive_scan);
      processor.splitConnected(cluster_dist_euclid_);
      processor.removeLessThan(min_points_per_cluster_);

      for (std::list<SampleSet*>::iterator i = processor.getClusters().begin(); i != processor.getClusters().end(); i++)
      {
        tf::Point cluster_position = (*i)->getPosition();

        for (int j = 0; j < gt_cluster_pose->poses.size(); j++)
        {
          // Only use clusters which are close to a "marker"
          double dist_x = gt_cluster_pose->poses[j].position.x - cluster_position[0],
                 dist_y = gt_cluster_pose->poses[j].position.y - cluster_position[1],
                 dist_abs = sqrt(dist_x * dist_x + dist_y * dist_y);
          if (dist_abs < 0.1)
          {
            data.push_back(processor.calcClusterFeatures(*i, *scan, feature_set_size_));
            break;
          }
        }
      }
      message_num++;
    }
    it_scans++;
    it_poses++;
  }
  bag.close();

  printf("\t Got %i scan messages, %i samples, from %s  \n", message_num, (int)data.size() - initial_pos_data_size,
         rosbag_file);
}

void ExtractFeatureMatrix::loadNegData(const char* rosbag_file, const char* scan_topic,
                                       std::vector<std::vector<float>>& data)
{
  rosbag::Bag bag;
  bag.open(rosbag_file, rosbag::bagmode::Read);
  std::vector<std::string> topics;
  topics.push_back(std::string(scan_topic));
  rosbag::View view(bag, rosbag::TopicQuery(topics));

  int message_num = 0;
  int initial_neg_data_size = (int)data.size();
  BOOST_FOREACH (rosbag::MessageInstance const m, view)
  {
    sensor_msgs::PointCloud::ConstPtr scan = m.instantiate<sensor_msgs::PointCloud>();
    if (scan != NULL)
    {
      sensor_msgs::PointCloud filtered_negative_scan;
      filtered_negative_scan.header = scan->header;
      filtered_negative_scan.channels.resize(scan->channels.size());

      sensor_msgs::PointCloud filtered_negative_scan_distance;
      filtered_negative_scan_distance.header = scan->header;
      filtered_negative_scan_distance.channels.resize(scan->channels.size());

      for (int i = 0; i != scan->channels.size(); i++)
      {
        filtered_negative_scan.channels[i].name = scan->channels[i].name;
        filtered_negative_scan_distance.channels[i].name = scan->channels[i].name;
      }

      otsu_filter::OtsuFilter filter;
      filter.filterRadar(*scan, filtered_negative_scan_distance, range_min_, range_max_, -100 * PI / 180,
                         -260 * PI / 180);
      if (filtered_negative_scan_distance.points.size() == 0)
        break;
      int min_intensity = filter.getOtsuThreshold(&filtered_negative_scan_distance, 0);
      filter.filterRadarIntensity(filtered_negative_scan_distance, filtered_negative_scan, min_intensity - 20);

      // Radar Processor GNN
      HumanDetector processor(filtered_negative_scan);
      processor.splitConnected(cluster_dist_euclid_);
      processor.removeLessThan(min_points_per_cluster_);

      for (std::list<SampleSet*>::iterator i = processor.getClusters().begin(); i != processor.getClusters().end(); i++)
      {
        if (rand() % undersample_negative_factor_ == 0)  // one way of undersampling the negative class
          data.push_back(processor.calcClusterFeatures(*i, *scan, feature_set_size_));
      }
      message_num++;
    }
  }
  bag.close();

  printf("\t Got %i scan messages, %i samples, from %s  \n", message_num, (int)data.size() - initial_neg_data_size,
         rosbag_file);
}

void ExtractFeatureMatrix::saveCvMatrix(const std::vector<std::vector<float>>& data)
{
  size_t sample_size = data.size();
  size_t feature_dim = data[0].size();

  CvMat* cv_data = cvCreateMat(sample_size, feature_dim, CV_32FC1);

  int j = 0;
  for (std::vector<std::vector<float>>::const_iterator i = data.begin(); i != data.end(); i++)
  {
    float* data_row = (float*)(cv_data->data.ptr + cv_data->step * j);
    for (int k = 0; k < feature_dim; k++)
      data_row[k] = (*i)[k];

    j++;
  }

  std::cout << "Saving data into: " << save_file_path_ << std::endl;
  cv::FileStorage fs(save_file_path_, cv::FileStorage::WRITE);
  fs << "feature_matrix" << cv::cvarrToMat(cv_data);
  fs.release();
}
} 
}  // namespace radar_human_tracker
