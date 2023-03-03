#include "otsu_filter.h"

namespace multi_posture_leg_tracker {
namespace otsu_filter
{
OtsuFilter::OtsuFilter()
{
  loadParameters();
  pc_publisher = nh.advertise<sensor_msgs::PointCloud>(pc_publisher_topic, 1);
  pc_subscriber = nh.subscribe(pc_subscriber_topic, 1, &OtsuFilter::filterPointCloud, this);
}

sensor_msgs::PointCloud OtsuFilter::message2Pointcloud(const sensor_msgs::PointCloudConstPtr& msg)
{
  sensor_msgs::PointCloud pc;
  pc.header = msg->header;
  pc.points = msg->points;
  pc.channels.resize(msg->channels.size());

  for (int i = 0; i != msg->channels.size(); i++)
  {
    pc.channels[i].values = msg->channels[i].values;
  }
  return pc;
}

void OtsuFilter::loadParameters()
{
  std::string pre_path = ros::this_node::getName();

  if (!nh.getParam(pre_path + "/dual_mode", dual_mode))
    std::cout << "Parameter Error: dual_mode, Using default: false" << std::endl;
  if (!nh.getParam(pre_path + "/pc_publisher_topic", pc_publisher_topic))
    std::cout << "Parameter Error: pc_publisher_topic" << std::endl;
  if (!nh.getParam(pre_path + "/pc_subscriber_topic", pc_subscriber_topic))
    std::cout << "Parameter Error: pc_subscriber_topic" << std::endl;
  if (!nh.getParam(pre_path + "/ch_angle", ch_angle))
    std::cout << "Parameter Error: ch_angle" << std::endl;
  if (!nh.getParam(pre_path + "/ch_range", ch_range))
    std::cout << "Parameter Error: ch_range" << std::endl;
  if (!nh.getParam(pre_path + "/min_range", min_range))
    std::cout << "Parameter Error: min_range" << std::endl;
  if (!nh.getParam(pre_path + "/max_range", max_range))
    std::cout << "Parameter Error: max_range" << std::endl;
  if (!nh.getParam(pre_path + "/min_angle", min_angle))
    std::cout << "Parameter Error: min_angle" << std::endl;
  if (!nh.getParam(pre_path + "/max_angle", max_angle))
    std::cout << "Parameter Error: max_angle" << std::endl;

  if (dual_mode)
  {
    ch_angle_dual = ch_angle + 2;
    ch_range_dual = ch_range + 2;
  }
  else
  {
    ch_angle_dual = ch_angle;
    ch_range_dual = ch_range;
  }
}

void OtsuFilter::filterRadar(const sensor_msgs::PointCloud& scan, sensor_msgs::PointCloud& filtered_scan,
                             float min_range, float max_range, float min_angle, float max_angle)
{
  for (int i = 0; i != scan.points.size(); i++)
  {
    if (scan.channels[2].values[i] > min_range && scan.channels[2].values[i] < max_range &&
        (scan.channels[1].values[i] > min_angle || max_angle > scan.channels[1].values[i]))
    {
      filtered_scan.channels[0].values.push_back(scan.channels[0].values[i]);
      filtered_scan.channels[1].values.push_back(scan.channels[1].values[i]);
      filtered_scan.channels[2].values.push_back(scan.channels[2].values[i]);
      filtered_scan.points.push_back(scan.points[i]);
    }
  }
}

void OtsuFilter::filterPointCloud(const sensor_msgs::PointCloudConstPtr& msg)
{
  sensor_msgs::PointCloud scan = message2Pointcloud(msg);
  sensor_msgs::PointCloud filtered_scan;
  sensor_msgs::PointCloud filtered_scan_distance;
  filtered_scan.header = scan.header;
  filtered_scan.channels.resize(scan.channels.size());
  filtered_scan_distance.header = scan.header;
  filtered_scan_distance.channels.resize(scan.channels.size());

  for (int i = 0; i != msg->channels.size(); i++)
  {
    filtered_scan.channels[i].name = msg->channels[i].name;
    filtered_scan_distance.channels[i].name = msg->channels[i].name;
  }

  if (scan.points.size() == 0)
    return;

  filterRadar(scan, filtered_scan_distance, min_range, max_range, min_angle, max_angle);
  int intensity_min = getOtsuThreshold(&filtered_scan_distance, 0);
  filterRadarIntensity(filtered_scan_distance, filtered_scan, intensity_min - 20);

  ROS_DEBUG_STREAM("Raw scan size: " << scan.points.size());
  ROS_DEBUG_STREAM("Filtered scan size: " << filtered_scan.points.size());

  pc_publisher.publish(filtered_scan);
}

void OtsuFilter::filterRadarIntensity(const sensor_msgs::PointCloud& scan, sensor_msgs::PointCloud& filtered_scan,
                                      int intensity_min)
{
  for (size_t i = 0; i != scan.points.size(); i++)
  {
    if (scan.channels[0].values[i] > intensity_min)
    {
      filtered_scan.channels[0].values.push_back(scan.channels[0].values[i]);
      filtered_scan.channels[2].values.push_back(scan.channels[2].values[i]);
      filtered_scan.channels[1].values.push_back(scan.channels[1].values[i]);
      filtered_scan.points.push_back(scan.points[i]);
    }
  }
}

int OtsuFilter::getOtsuThreshold(sensor_msgs::PointCloud* scan, int var_identifier)
{
  // Get Boundaries
  int min_int = scan->channels[var_identifier].values[0];
  int max_int = scan->channels[var_identifier].values[0];
  float mean_int = scan->channels[var_identifier].values[0];

  for (int i = 1; i != scan->points.size(); i++)
  {
    if (min_int > scan->channels[var_identifier].values[i])
      min_int = scan->channels[var_identifier].values[i];

    if (max_int < scan->channels[var_identifier].values[i])
      max_int = scan->channels[var_identifier].values[i];

    mean_int += scan->channels[var_identifier].values[i];
  }

  mean_int /= (float)scan->points.size();

  // Get Histogram
  std::vector<int> histo;
  histo.resize(max_int - min_int, 0);

  for (int i = 0; i != scan->points.size(); i++)
    histo[scan->channels[var_identifier].values[i] - min_int]++;

  // Hill Climbing
  bool use_break = false;
  int hill_climbing_max = 100;
  int max_rando = (int)(histo.size() / 7.);

  int threshold = (int)((max_int - min_int) / 2.);
  float otsu_gradient = 0;

  for (int i = 0; i != hill_climbing_max; i++)
  {
    bool break_hill_climbing = true;
    float temp_best_otsu_gradient = 0;

    if (threshold == 0)
    {
      int increment = rand() % max_rando + 1;
      temp_best_otsu_gradient = getOtsuGradient(&histo, threshold + increment);
      if (temp_best_otsu_gradient > otsu_gradient)
      {
        otsu_gradient = temp_best_otsu_gradient;
        threshold += increment;
        break_hill_climbing = false;
      }
    }
    else if (threshold == histo.size() - 1)
    {
      int increment = rand() % max_rando + 1;
      temp_best_otsu_gradient = getOtsuGradient(&histo, threshold - increment);
      if (temp_best_otsu_gradient > otsu_gradient)
      {
        otsu_gradient = temp_best_otsu_gradient;
        threshold -= increment;
        break_hill_climbing = false;
      }
    }
    else
    {
      bool break_hill_climbing = true;

      int increment = 1;
      temp_best_otsu_gradient = getOtsuGradient(&histo, threshold - increment);
      if (temp_best_otsu_gradient > otsu_gradient)
      {
        otsu_gradient = temp_best_otsu_gradient;
        threshold -= increment;
        break_hill_climbing = false;
      }

      increment = 1;
      temp_best_otsu_gradient = getOtsuGradient(&histo, threshold + increment);
      if (temp_best_otsu_gradient > otsu_gradient)
      {
        otsu_gradient = temp_best_otsu_gradient;
        threshold += increment;
        break_hill_climbing = false;
      }
    }

    if (break_hill_climbing && use_break)
      break;
  }

  return threshold + min_int;
}

float OtsuFilter::getOtsuGradient(std::vector<int>* histogram, int threshold)
{
  float mu = 0;

  // Generate means and distribution values
  float p_0 = 0.;
  float mu_0 = 0.;
  for (int i = 0; i != threshold + 1; i++)
  {
    p_0 += histogram->at(i);
    mu_0 += i * histogram->at(i);
  }

  p_0 /= (float)histogram->size();

  float p_1 = 1. - p_0;
  float mu_1 = 0.;
  for (int i = threshold + 1; i != histogram->size(); i++)
  {
    mu_1 += i * histogram->at(i);
  }

  mu = mu_0 + mu_1;
  mu_0 /= (float)threshold;
  mu_1 /= (float)(histogram->size() - threshold);
  mu /= (float)(histogram->size() - threshold);

  // Generate variances in classes
  float s_0 = 0.;
  for (int i = 0; i != threshold + 1; i++)
    s_0 += (i - mu_0) * (i - mu_0) * histogram->at(i);

  s_0 /= (float)(histogram->size());

  float s_1 = 0.;
  for (int i = threshold + 1; i != histogram->size(); i++)
    s_1 += (i - mu_1) * (i - mu_1) * histogram->at(i);

  s_1 /= (float)(histogram->size());

  // Compute variances between classes
  float omega_inter = (p_0 * (mu_0 - mu) * (mu_0 - mu)) + (p_1 * (mu_1 - mu) * (mu_1 - mu));
  float omega_in = (p_0 * s_0 * s_0) + (p_1 * s_1 * s_1);

  // Return gradient
  return omega_inter / omega_in;
}
}  // namespace otsu_filter
}
