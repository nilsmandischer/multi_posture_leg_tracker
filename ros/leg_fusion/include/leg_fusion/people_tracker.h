#ifndef PEDESTRIANLOCALISATION_H
#define PEDESTRIANLOCALISATION_H

#include <ros/ros.h>
#include <ros/time.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>
#include <rc_tracking_msgs/Person.h>
#include <rc_tracking_msgs/PersonArray.h>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <bayes_tracking/BayesFilter/bayesFlt.hpp>

#include <XmlRpcValue.h>

#include <string.h>
#include <vector>
#include <math.h>

#include "leg_fusion/simple_tracking.h"
#include "leg_fusion/asso_exception.h"

#define INVALID_ID -1  // For publishing trajectories

namespace multi_posture_leg_tracker {
class PeopleTracker
{
public:
  PeopleTracker();

private:
  /**
   * @brief publishTrajectory
   * Create and publish trajectories
   */
  void publishTrajectory(std::vector<geometry_msgs::Pose> poses, std::vector<geometry_msgs::Pose> vels,
                         std::vector<geometry_msgs::Pose> vars, std::vector<long> pids, ros::Publisher& pub);

  /**
   * @brief createVisualisation
   * Visualize markers in rviz
   */
  void createVisualisation(ros::Time now, std::vector<geometry_msgs::Pose> points, std::vector<long> pids,
                           ros::Publisher& pub);

  /**
   * @brief cartesianToPolar
   * Transform Cartesian coordinates to Polar coordinates
   */
  std::vector<double> cartesianToPolar(geometry_msgs::Point point);

  /**
   * @brief detectorCallback
   * Callback containing main logic of the tracker is executed every time detections are received.
   */
  void detectorCallback(const geometry_msgs::PoseArray::ConstPtr& pta, string detector);

  /**
   * @brief connectCallback
   * Connect callback function and subscribe detection topics
   */
  void connectCallback(ros::NodeHandle& n);

  /**
   * @brief parseParams
   * Load parameters from the server
   */
  void parseParams(ros::NodeHandle);

  std::string generateUUID(std::string time, long id)
  {
    boost::uuids::name_generator gen(dns_namespace_uuid);
    time += num_to_str<long>(id);

    return num_to_str<boost::uuids::uuid>(gen(time.c_str()));
  }

  template <typename T>
  std::string num_to_str(T num)
  {
    std::stringstream ss;
    ss << num;
    return ss.str();
  }

  ros::Publisher pub_trajectory;           /**< trajectory publisher */
  ros::Publisher pub_marker;               /**< marker publisher */
  ros::Publisher pub_tracked_person_array; /**< Person tracks publisher */
  tf::TransformListener* listener;
  std::string target_frame; /**< frame where tracks will be published */

  bool publish_occluded; /**< whether publish occluded tracks without matched detections */
  double startup_time;   /**< time when the tracker starts up, used for ID generalization */
  std::string startup_time_str;

  boost::uuids::uuid dns_namespace_uuid;

  SimpleTracking<EKFilter>* ekf = NULL;
  SimpleTracking<UKFilter>* ukf = NULL;
  SimpleTracking<PFilter>* pf = NULL;
  std::map<std::pair<std::string, std::string>, ros::Subscriber> subscribers;
  std::vector<boost::tuple<long, geometry_msgs::Pose, geometry_msgs::Pose, geometry_msgs::Pose> >
      previous_poses; /**< Used for trajectory creation */
};
}
#endif  // PEDESTRIANLOCALISATION_H
