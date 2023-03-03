#include <ros/ros.h>
#include "local_occupancy_grid_mapping_node/local_occupancy_grid_mapping.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "occupancy_grid_mapping");

  ros::NodeHandle nh("~");
  std::string scan_topic;
  nh.param("scan_topic", scan_topic, std::string("scan"));
  multi_posture_leg_tracker::laser_human_tracker::OccupancyGridMapping ogm(nh, scan_topic);

  ros::spin();
  return 0;
}
