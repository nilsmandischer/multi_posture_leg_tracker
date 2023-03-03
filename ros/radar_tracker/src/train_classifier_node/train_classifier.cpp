#include "train_classifier.h"

namespace multi_posture_leg_tracker {
namespace radar_human_tracker
{

TrainClassifier::TrainClassifier(ros::NodeHandle& nh) : nh_(nh)
{
  // Get ROS param
  if (!nh_.getParam("mode", mode_))
    ROS_ERROR("Couldn't get mode from ros param server");
  if (!nh_.getParam("classifier_type", classifier_type_))
    ROS_ERROR("Couldn't get classifier_type from ros param server");
  if (!nh_.getParam("model_file", model_file_))
    ROS_ERROR("Couldn't get model_file from ros param server");
  if (!nh_.getParam("feature_set_size", feature_set_size_))
    ROS_ERROR("Couldn't get feature_set_size from ros param server");

  if (!nh_.getParam("cluster_dist_euclid", cluster_dist_euclid_))
    ROS_ERROR("Couldn't get cluster_dist_euclid from ros param server");
  if (!nh_.getParam("min_points_per_cluster", min_points_per_cluster_))
    ROS_ERROR("Couldn't get min_points_per_cluster from ros param server");

  if (!nh_.getParam("undersample_negative_factor", undersample_negative_factor_))
    ROS_ERROR("Couldn't get undersample_negative_factor from ros param server");

  if (!nh_.getParam("range_min", range_min_))
    ROS_ERROR("Couldn't get range_min from ros param server");
  if (!nh_.getParam("range_max", range_max_))
    ROS_ERROR("Couldn't get range_max from ros param server");

  if (mode_ != 0 and mode_ != 1)
  {
    ROS_ERROR("The mode should be 0 OR 1! 0 for train, 1 for test.");
    exit(1);
  }

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
}

void TrainClassifier::loadData(int argc, char** argv)
{
  // Parse command line arguements and load data
  printf("\nLoading data...\n");
  for (int i = 0; i < argc; i++)
  {
    switch (mode_)
    {
      case 0:  // train and test
        if (!strcmp(argv[i], "--leg"))
        {
          char* rosbag_file = argv[++i];
          char* scan_topic = argv[++i];
          char* cluster_topic = argv[++i];
          loadAnnotatedData(rosbag_file, scan_topic, cluster_topic, train_leg_data_);
        }
        else if (!strcmp(argv[i], "--squat"))
        {
          char* rosbag_file = argv[++i];
          char* scan_topic = argv[++i];
          char* cluster_topic = argv[++i];
          loadAnnotatedData(rosbag_file, scan_topic, cluster_topic, train_squat_data_);
        }
        else if (!strcmp(argv[i], "--neg"))
        {
          char* rosbag_file = argv[++i];
          char* scan_topic = argv[++i];
          loadNegData(rosbag_file, scan_topic, train_neg_data_);
        }
        else if (!strcmp(argv[i], "--neg_annotated"))
        {
          char* rosbag_file = argv[++i];
          char* scan_topic = argv[++i];
          char* cluster_topic = argv[++i];
          loadAnnotatedData(rosbag_file, scan_topic, cluster_topic, train_neg_data_);
        }
        else if (!strcmp(argv[i], "--test_leg"))
        {
          char* rosbag_file = argv[++i];
          char* scan_topic = argv[++i];
          char* cluster_topic = argv[++i];
          loadAnnotatedData(rosbag_file, scan_topic, cluster_topic, test_leg_data_);
        }
        else if (!strcmp(argv[i], "--test_squat"))
        {
          char* rosbag_file = argv[++i];
          char* scan_topic = argv[++i];
          char* cluster_topic = argv[++i];
          loadAnnotatedData(rosbag_file, scan_topic, cluster_topic, test_squat_data_);
        }
        else if (!strcmp(argv[i], "--test_neg"))
        {
          char* rosbag_file = argv[++i];
          char* scan_topic = argv[++i];
          loadNegData(rosbag_file, scan_topic, test_neg_data_);
        }
        else if (!strcmp(argv[i], "--test_neg_annotated"))
        {
          char* rosbag_file = argv[++i];
          char* scan_topic = argv[++i];
          char* cluster_topic = argv[++i];
          loadAnnotatedData(rosbag_file, scan_topic, cluster_topic, test_neg_data_);
        }
        // load pre-extracted feature matrix from .xml files directly
        else if (!strcmp(argv[i], "--leg_features"))
        {
          char* feature_matrix_file = argv[++i];
          loadCvFeatureMatrix(feature_matrix_file, train_leg_data_);
        }
        else if (!strcmp(argv[i], "--squat_features"))
        {
          char* feature_matrix_file = argv[++i];
          loadCvFeatureMatrix(feature_matrix_file, train_squat_data_);
        }
        else if (!strcmp(argv[i], "--neg_features"))
        {
          char* feature_matrix_file = argv[++i];
          loadCvFeatureMatrix(feature_matrix_file, train_neg_data_);
        }
        else if (!strcmp(argv[i], "--test_leg_features"))
        {
          char* feature_matrix_file = argv[++i];
          loadCvFeatureMatrix(feature_matrix_file, test_leg_data_);
        }
        else if (!strcmp(argv[i], "--test_squat_features"))
        {
          char* feature_matrix_file = argv[++i];
          loadCvFeatureMatrix(feature_matrix_file, test_squat_data_);
        }
        else if (!strcmp(argv[i], "--test_neg_features"))
        {
          char* feature_matrix_file = argv[++i];
          loadCvFeatureMatrix(feature_matrix_file, test_neg_data_);
        }
        else if (!strcmp(argv[i], "--save_file_path"))
        {
          save_file_path_ = std::string(argv[++i]);
        }
        break;
      case 1:  // only test
        if (!strcmp(argv[i], "--test_leg"))
        {
          char* rosbag_file = argv[++i];
          char* scan_topic = argv[++i];
          char* cluster_topic = argv[++i];
          loadAnnotatedData(rosbag_file, scan_topic, cluster_topic, test_leg_data_);
        }
        else if (!strcmp(argv[i], "--test_squat"))
        {
          char* rosbag_file = argv[++i];
          char* scan_topic = argv[++i];
          char* cluster_topic = argv[++i];
          loadAnnotatedData(rosbag_file, scan_topic, cluster_topic, test_squat_data_);
        }
        else if (!strcmp(argv[i], "--test_neg"))
        {
          char* rosbag_file = argv[++i];
          char* scan_topic = argv[++i];
          loadNegData(rosbag_file, scan_topic, test_neg_data_);
        }
        else if (!strcmp(argv[i], "--test_neg_annotated"))
        {
          char* rosbag_file = argv[++i];
          char* scan_topic = argv[++i];
          char* cluster_topic = argv[++i];
          loadAnnotatedData(rosbag_file, scan_topic, cluster_topic, test_neg_data_);
        }
        else if (!strcmp(argv[i], "--test_leg_features"))
        {
          char* feature_matrix_file = argv[++i];
          loadCvFeatureMatrix(feature_matrix_file, test_leg_data_);
        }
        else if (!strcmp(argv[i], "--test_squat_features"))
        {
          char* feature_matrix_file = argv[++i];
          loadCvFeatureMatrix(feature_matrix_file, test_squat_data_);
        }
        else if (!strcmp(argv[i], "--test_neg_features"))
        {
          char* feature_matrix_file = argv[++i];
          loadCvFeatureMatrix(feature_matrix_file, test_neg_data_);
        }
        else if (!strcmp(argv[i], "--save_file_path"))
        {
          save_file_path_ = std::string(argv[++i]);
        }
        break;
    }
  }

  // Check we have a valid save file
  if (save_file_path_.empty())
  {
    ROS_ERROR("Save file not specified properly in command line arguments \nExiting");
    exit(1);
  }

  printf("\n  Total leg training samples: %i \t Total negative training "
         "samples: %i \t Total squat training samples: %i \n",
         (int)train_leg_data_.size(), (int)train_neg_data_.size(), (int)train_squat_data_.size());
  printf("  Total leg test samples: %i \t Total negative test samples: "
         "%i \t Total squat test samples: %i \n\n",
         (int)test_leg_data_.size(), (int)test_neg_data_.size(), (int)test_squat_data_.size());
}

void TrainClassifier::run()
{
  switch (mode_)
  {
    case 0:
    {
      if (train_leg_data_.empty() or train_neg_data_.empty() or train_squat_data_.empty())
      {
        ROS_ERROR("Data not loaded from rosbags properly \nExiting");
        exit(1);
      }
      printf("Training classifier... \n\n");
      classifier_->train(train_leg_data_, train_squat_data_, train_neg_data_, save_file_path_ + "eval/");
      std::string save_model_file_path = save_file_path_ + classifier_type_ + "_" + std::to_string(feature_set_size_);
      classifier_->saveModel(save_model_file_path);
      printf("Train done! \n\n");
      if (!test_leg_data_.empty() or !test_neg_data_.empty() or !test_squat_data_.empty())
      {
        printf("Testing classifier... \n\n");
        std::string save_test_file = save_file_path_ + "eval/" + classifier_type_.c_str() + "_" +
                                     std::to_string(feature_set_size_) + "_test.yaml";
        classifier_->test(test_leg_data_, test_squat_data_, test_neg_data_, save_test_file);
      }
      break;
    }

    case 1:
    {
      classifier_->loadModel(model_file_);

      int feature_set[6] = { FEATURE_SET_0, FEATURE_SET_1, FEATURE_SET_2, FEATURE_SET_3, FEATURE_SET_4 };
      if (classifier_->getFeatureSetSize() != feature_set[feature_set_size_])
      {
        ROS_ERROR("The given param feature_set_size %d doesn't correspond to the loaded model %d. Please check again !",
                  feature_set[feature_set_size_], classifier_->getFeatureSetSize());
        exit(1);
      }

      printf("Testing classifier... \n\n");
      std::string save_test_file =
          save_file_path_ + "eval/" + classifier_type_.c_str() + "_" + std::to_string(feature_set_size_) + "_test.yaml";
      classifier_->test(test_leg_data_, test_squat_data_, test_neg_data_, save_test_file);
      break;
    }
  }
}

void TrainClassifier::loadAnnotatedData(const char* rosbag_file, const char* scan_topic, const char* cluster_topic,
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

void TrainClassifier::loadNegData(const char* rosbag_file, const char* scan_topic,
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

void TrainClassifier::loadCvFeatureMatrix(const char* file, std::vector<std::vector<float>>& data)
{
  std::cout << "Loading data from: " << file << std::endl;
  cv::FileStorage fs;
  fs.open(file, cv::FileStorage::READ);
  if (!fs.isOpened())
  {
    ROS_ERROR_STREAM("Failed to open " << file);
    exit(1);
  }

  cv::Mat feature_matrix;
  fs["feature_matrix"] >> feature_matrix;
  for (int i = 0; i < feature_matrix.rows; i++)
  {
    std::vector<float> feature(feature_matrix.cols);
    for (int j = 0; j < feature_matrix.cols; j++)
    {
      feature[j] = feature_matrix.at<float>(i, j);
    }
    data.push_back(feature);
  }

  std::cout << "Finished loading data from: " << file << std::endl;
}
}  // namespace radar_human_tracker
}
