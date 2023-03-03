#include <ros/ros.h>
#include "human_tracker_node/human_tracker.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "radar_human_tracker");
  ros::AsyncSpinner spinner(0);
  spinner.start();
  multi_posture_leg_tracker::radar_human_tracker::HumanTracker human_tracker;
  ros::waitForShutdown();
}
