#include <ros/ros.h>
#include "human_detector_node/human_detector.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "laser_human_detector");
  ros::NodeHandle nh;
  multi_posture_leg_tracker::laser_human_tracker::HumanDetector detector(nh);
  ros::spin();
  return 0;
}
