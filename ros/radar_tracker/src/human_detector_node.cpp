#include "ros/ros.h"
#include "human_detector_node/human_detector.h"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "radar_human_detector");
  multi_posture_leg_tracker::radar_human_tracker::HumanDetector detector;
  ros::spin();
}
