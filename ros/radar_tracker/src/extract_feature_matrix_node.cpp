#include <ros/ros.h>
#include "extract_feature_matrix_node/extract_feature_matrix.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "radar_extract_feature_matrix_node");
  ros::NodeHandle nh;

  multi_posture_leg_tracker::radar_human_tracker::ExtractFeatureMatrix efm(nh);
  efm.loadData(argc, argv);
  efm.run();

  printf("Finished successfully!\n");
  return 0;
}
