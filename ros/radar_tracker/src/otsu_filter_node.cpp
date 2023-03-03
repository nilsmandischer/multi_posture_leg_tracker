#include "otsu_filter_node/otsu_filter.h"
#include "ros/ros.h"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "otsu_filter_node");
  multi_posture_leg_tracker::otsu_filter::OtsuFilter filter;
  ros::spin();
}
