#include "train_classifier.h"

namespace multi_posture_leg_tracker {

namespace laser_human_tracker
{
TrainClassifier::TrainClassifier(ros::NodeHandle nh)
{
  // Get ROS params (all have default values so it's not critical we get them
  // all)
  nh.param("cluster_dist_euclid", cluster_dist_euclid_, 0.13);
  nh.param("min_points_per_cluster", min_points_per_cluster_, 3);
  nh.param("undersample_negative_factor", undersample_negative_factor_, 50);
  nh.param("feature_set_size", feature_set_size_, 0);
  nh.param("classifier_type", classifier_type_, std::string("opencv_random_forest"));
  nh.param("mode", mode_, 0);  // 0 for train; 1 for test
  nh.param("model_file", model_file_, std::string(""));

  // Print back params:
  printf("\nROS parameters: \n");
  printf("mode: %d \n", mode_);
  printf("classifier_type: %s \n", classifier_type_.c_str());
  printf("feature_set_size: %d \n", feature_set_size_);
  printf("cluster_dist_euclid:%.2fm \n", cluster_dist_euclid_);
  printf("min_points_per_cluster:%i \n", min_points_per_cluster_);
  printf("undersample_negative_factor:%i \n", undersample_negative_factor_);
  printf("\n");

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
    if (mode_ == 0)
    {
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
      else if (!strcmp(argv[i], "--save_file_path"))
      {
        save_file_path_ = std::string(argv[++i]);
      }
    }
    else  // mode_ = 1
    {
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
      else if (!strcmp(argv[i], "--save_file_path"))
      {
        save_file_path_ = std::string(argv[++i]);
      }
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

      int feature_set[7] = { FEATURE_SET_0, FEATURE_SET_1, FEATURE_SET_2, FEATURE_SET_3, FEATURE_SET_4 };
      if (classifier_->getFeatureSetSize() != feature_set[feature_set_size_])
      {
        ROS_ERROR("The given param feature_set_size doesn't corresponding to the loaded model. Please check again!");
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
  int initial_data_size = (int)data.size();

  rosbag::Bag bag;
  bag.open(rosbag_file, rosbag::bagmode::Read);

  std::vector<sensor_msgs::LaserScan::ConstPtr> scans;
  std::vector<geometry_msgs::PoseArray::ConstPtr> ground_truth_cluster_poses;

  rosbag::View view_1(bag, rosbag::TopicQuery(std::string(scan_topic)));
  BOOST_FOREACH (rosbag::MessageInstance const m, view_1)  // sort by time?
  {
    sensor_msgs::LaserScan::ConstPtr scan = m.instantiate<sensor_msgs::LaserScan>();
    scans.push_back(scan);
  }

  rosbag::View view_2(bag, rosbag::TopicQuery(std::string(cluster_topic)));
  BOOST_FOREACH (rosbag::MessageInstance const m, view_2)
  {
    geometry_msgs::PoseArray::ConstPtr pose_array_msg = m.instantiate<geometry_msgs::PoseArray>();
    ground_truth_cluster_poses.push_back(pose_array_msg);
  }

  std::vector<sensor_msgs::LaserScan::ConstPtr>::iterator it_scans = scans.begin();
  std::vector<geometry_msgs::PoseArray::ConstPtr>::iterator it_poses = ground_truth_cluster_poses.begin();
  while (it_scans != scans.end())
  {
    sensor_msgs::LaserScan::ConstPtr scan = *it_scans;
    geometry_msgs::PoseArray::ConstPtr gt_cluster_pose = *it_poses;

    if (scan->ranges.size() and gt_cluster_pose->poses.size())
    {
      laser_processor::ScanProcessor processor(*scan);
      processor.splitConnected(cluster_dist_euclid_);
      processor.removeLessThan(min_points_per_cluster_);
      double min_dist = std::numeric_limits<double>::infinity();

      for (std::list<laser_processor::SampleSet*>::iterator i = processor.getClusters().begin();
           i != processor.getClusters().end(); i++)
      {
        tf::Point cluster_position = (*i)->getPosition();

        for (int j = 0; j < gt_cluster_pose->poses.size(); j++)
        {
          // Only use clusters which are close to a "marker"
          double dist_x = gt_cluster_pose->poses[j].position.x - cluster_position[0],
                 dist_y = gt_cluster_pose->poses[j].position.y - cluster_position[1],
                 dist_abs = sqrt(dist_x * dist_x + dist_y * dist_y);

          if (dist_abs < min_dist)
          {
            min_dist = dist_abs;
          }

          if (dist_abs < 0.1)
          {
            data.push_back(cf_.calcClusterFeatures(*i, *scan, feature_set_size_));
            break;
          }
        }
      }
      if (min_dist >= 0.1)
        std::cout << "min dist: " << min_dist << std::endl;
      message_num++;
    }
    it_scans++;
    it_poses++;
  }
  bag.close();

  printf("\t Got %i scan messages, %i samples, from %s  \n", message_num, (int)data.size() - initial_data_size,
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
    sensor_msgs::LaserScan::ConstPtr scan = m.instantiate<sensor_msgs::LaserScan>();
    if (scan != NULL)
    {
      laser_processor::ScanProcessor processor(*scan);
      processor.splitConnected(cluster_dist_euclid_);
      processor.removeLessThan(min_points_per_cluster_);

      for (std::list<laser_processor::SampleSet*>::iterator i = processor.getClusters().begin();
           i != processor.getClusters().end(); i++)
      {
        if (rand() % undersample_negative_factor_ == 0)  // one way of undersampling the negative class
          data.push_back(cf_.calcClusterFeatures(*i, *scan, feature_set_size_));
      }
      message_num++;
    }
  }
  bag.close();

  printf("\t Got %i scan messages, %i samples, from %s  \n", message_num, (int)data.size() - initial_neg_data_size,
         rosbag_file);
}
}  // namespace laser_human_tracker
}