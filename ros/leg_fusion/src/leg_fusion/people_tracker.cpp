#include "leg_fusion/people_tracker.h"
#include <tf/transform_datatypes.h>
#include <XmlRpc.h>

namespace multi_posture_leg_tracker {

PeopleTracker::PeopleTracker()
{
  ros::NodeHandle n;

  listener = new tf::TransformListener();

  startup_time = ros::Time::now().toSec();
  startup_time_str = num_to_str<double>(startup_time);

  // Declare variables that can be modified by launch file or command line.
  std::string pub_topic_trajectory;
  std::string pub_marker_topic;
  std::string pub_topic_tracked_person_array;

  // Initialize node parameters from launch file or command line.
  // Use a private node handle so that multiple instances of the node can be run simultaneously
  // while using different parameters.
  ros::NodeHandle private_node_handle("~");
  private_node_handle.param("target_frame", target_frame, std::string("/base_link"));
  private_node_handle.param("publish_occluded", publish_occluded, false);
  parseParams(private_node_handle);

  // Create a status callback.
  ros::SubscriberStatusCallback con_cb = boost::bind(&PeopleTracker::connectCallback, this, boost::ref(n));

  private_node_handle.param("trajectory", pub_topic_trajectory, std::string("/people_tracker/trajectory"));
  pub_trajectory = n.advertise<geometry_msgs::PoseArray>(pub_topic_trajectory.c_str(), 100, con_cb, con_cb);
  private_node_handle.param("marker", pub_marker_topic, std::string("/people_tracker/marker_array"));
  pub_marker = n.advertise<visualization_msgs::MarkerArray>(pub_marker_topic.c_str(), 100, con_cb, con_cb);
  private_node_handle.param("tracked_person_array", pub_topic_tracked_person_array,
                            std::string("/people_tracked"));
  pub_tracked_person_array =
      n.advertise<rc_tracking_msgs::PersonArray>(pub_topic_tracked_person_array.c_str(), 100, con_cb, con_cb);

  ros::spin();
}

void PeopleTracker::parseParams(ros::NodeHandle n)
{
  std::string filter;
  n.getParam("filter_type", filter);
  ROS_INFO_STREAM("Found filter type: " << filter);

  float stdLimit = 1.0;
  if (n.hasParam("std_limit"))
  {
    n.getParam("std_limit", stdLimit);
    ROS_INFO_STREAM("std_limit pruneTracks with " << stdLimit);
  }

  bool prune_named = false;
  if (n.hasParam("prune_named"))
  {
    n.getParam("prune_named", prune_named);
    ROS_INFO_STREAM("prune_named with " << prune_named);
  }

  if (filter == "EKF")
  {
    ekf = new SimpleTracking<EKFilter>(stdLimit, prune_named);
  }
  else if (filter == "UKF")
  {
    ukf = new SimpleTracking<UKFilter>(stdLimit, prune_named);
  }
  else if (filter == "PF")
  {
    pf = new SimpleTracking<PFilter>(stdLimit, prune_named);
  }
  else
  {
    ROS_FATAL_STREAM("Filter type " << filter
                                    << " is not specified. Unable to create the tracker. Please use either EKF, UKF or "
                                       "PF.");
    return;
  }

  XmlRpc::XmlRpcValue cv_noise;
  n.getParam("cv_noise_params", cv_noise);
  ROS_ASSERT(cv_noise.getType() == XmlRpc::XmlRpcValue::TypeStruct);
  ROS_INFO_STREAM("Constant Velocity Model noise: " << cv_noise);
  if (ekf != NULL)
  {
    ekf->createConstantVelocityModel(cv_noise["x"], cv_noise["y"]);
  }
  else if (ukf != NULL)
  {
    ukf->createConstantVelocityModel(cv_noise["x"], cv_noise["y"]);
  }
  else if (pf != NULL)
  {
    pf->createConstantVelocityModel(cv_noise["x"], cv_noise["y"]);
  }
  else
  {
    ROS_FATAL_STREAM("no filter configured.");
  }
  ROS_INFO_STREAM("Created " << filter << " based tracker using constant velocity prediction model.");

  XmlRpc::XmlRpcValue detectors;
  n.getParam("detectors", detectors);
  ROS_ASSERT(detectors.getType() == XmlRpc::XmlRpcValue::TypeStruct);
  for (XmlRpc::XmlRpcValue::ValueStruct::const_iterator it = detectors.begin(); it != detectors.end(); ++it)
  {
    ROS_INFO_STREAM("Found detector: " << (std::string)(it->first) << " ==> " << detectors[it->first]);
    observ_model_t om_flag;
    double pos_noise_x = .2;
    double pos_noise_y = .2;
    int seq_size = 5;
    double seq_time = 0.2;
    association_t association = NN;
    om_flag = CARTESIAN;

    try
    {
      if (detectors[it->first].hasMember("seq_size"))
        seq_size = (int)detectors[it->first]["seq_size"];
      if (detectors[it->first].hasMember("seq_time"))
        seq_time = (double)detectors[it->first]["seq_time"];
      if (detectors[it->first].hasMember("matching_algorithm"))
        association =
            detectors[it->first]["matching_algorithm"] == "NN" ?
                NN :
                detectors[it->first]["matching_algorithm"] == "NNJPDA" ?
                NNJPDA :
                detectors[it->first]["matching_algorithm"] == "NN_LABELED" ? NN_LABELED : throw(asso_exception());
      if (detectors[it->first].hasMember("cartesian_noise_params"))
      {  // legacy support
        pos_noise_x = detectors[it->first]["cartesian_noise_params"]["x"];
        pos_noise_y = detectors[it->first]["cartesian_noise_params"]["y"];
      }
      if (detectors[it->first].hasMember("noise_params"))
      {
        pos_noise_x = detectors[it->first]["noise_params"]["x"];
        pos_noise_y = detectors[it->first]["noise_params"]["y"];
      }
    }
    catch (XmlRpc::XmlRpcException& e)
    {
      ROS_FATAL_STREAM("XmlRpc::XmlRpcException: '" << e.getMessage() << "'\n"
                                                    << "Failed to parse definition for '" << (std::string)(it->first)
                                                    << "'. Check your parameters.");
      throw(e);
    }

    try
    {
      if (ekf != NULL)
      {
        ekf->addDetectorModel(it->first, association, om_flag, pos_noise_x, pos_noise_y, seq_size, seq_time);
      }
      else if (ukf != NULL)
      {
        ukf->addDetectorModel(it->first, association, om_flag, pos_noise_x, pos_noise_y, seq_size, seq_time);
      }
      else if (pf != NULL)
      {
        pf->addDetectorModel(it->first, association, om_flag, pos_noise_x, pos_noise_y, seq_size, seq_time);
      }
    }
    catch (asso_exception& e)
    {
      ROS_FATAL_STREAM("" << e.what() << " " << detectors[it->first]["matching_algorithm"]
                          << " is not specified. Unable to add " << (std::string)(it->first)
                          << " to the tracker. Please use either NN or NNJPDA as association algorithms.");
      return;
    }
    catch (observ_exception& e)
    {
      ROS_FATAL_STREAM("" << e.what() << " " << detectors[it->first]["observation_model"]
                          << " is not specified. Unable to add " << (std::string)(it->first)
                          << " to the tracker. Please use either CARTESIAN or POLAR as observation models.");
      return;
    }
    ros::Subscriber sub;
    if (detectors[it->first].hasMember("topic"))
    {
      subscribers[std::pair<std::string, std::string>(it->first, detectors[it->first]["topic"])] = sub;
    }
  }
}

void PeopleTracker::publishTrajectory(std::vector<geometry_msgs::Pose> poses, std::vector<geometry_msgs::Pose> vels,
                                      std::vector<geometry_msgs::Pose> vars, std::vector<long> pids,
                                      ros::Publisher& pub)
{
  /*** find trajectories ***/
  for (int i = 0; i < previous_poses.size(); i++)
  {
    if (boost::get<0>(previous_poses[i]) != INVALID_ID)
    {
      bool last_pose = true;
      for (int j = 0; j < pids.size(); j++)
      {
        if (pids[j] == boost::get<0>(previous_poses[i]))
        {
          last_pose = false;
          // break;
        }
      }
      if (last_pose)
      {
        geometry_msgs::PoseArray trajectory;
        geometry_msgs::PoseArray velocity;
        geometry_msgs::PoseArray variance;
        trajectory.header.seq = boost::get<0>(previous_poses[i]);  // tracking ID
        trajectory.header.stamp = ros::Time::now();
        trajectory.header.frame_id = target_frame;  // will be reused by P-N experts
        for (int j = 0; j < previous_poses.size(); j++)
        {
          if (boost::get<0>(previous_poses[j]) == trajectory.header.seq)
          {
            trajectory.poses.push_back(boost::get<3>(previous_poses[j]));
            velocity.poses.push_back(boost::get<2>(previous_poses[j]));
            variance.poses.push_back(boost::get<1>(previous_poses[j]));
            boost::get<0>(previous_poses[j]) = INVALID_ID;
          }
        }
        pub.publish(trajectory);
        ROS_INFO_STREAM("[Fused People Tracker] trajectory ID = " << trajectory.header.seq
                                                                  << ", timestamp = " << trajectory.header.stamp
                                                                  << ", poses size = " << trajectory.poses.size());
      }
    }
  }

  /*** clean up ***/
  for (int i = 0; i < previous_poses.size(); i++)
  {
    if (boost::get<0>(previous_poses[i]) == INVALID_ID)
      previous_poses.erase(previous_poses.begin() + i);
  }

  for (int i = 0; i < poses.size(); i++)
  {
    previous_poses.push_back(boost::make_tuple(pids[i], vars[i], vels[i], poses[i]));
  }
}

void PeopleTracker::createVisualisation(ros::Time now, std::vector<geometry_msgs::Pose> poses, std::vector<long> pids,
                                        ros::Publisher& pub)
{
  ROS_DEBUG("Creating markers");
  visualization_msgs::MarkerArray marker_array;
  for (int i = 0; i < poses.size(); i++)
  {
    // Create Human Model
    visualization_msgs::Marker marker;
    marker.header.frame_id = target_frame;
    marker.header.stamp = now;
    marker.ns = "fused_track_body";
    marker.color.r = 1;
    marker.color.g = 0;
    marker.color.b = 0.8;
    marker.color.a = 1;
    marker.pose.position.x = poses[i].position.x;
    marker.pose.position.y = poses[i].position.y;
    marker.id = pids[i];
    marker.type = visualization_msgs::Marker::CYLINDER;
    marker.scale.x = 0.23;         // 0.2 0.15
    marker.scale.y = 0.23;         // 0.2 0.15
    marker.scale.z = 1.2;          // 1.2 0.8
    marker.pose.position.z = 0.8;  // 0.8 0.5
    marker.lifetime = ros::Duration(0.13);
    marker_array.markers.push_back(marker);

    // Sphere for head shape
    marker.ns = "fused_track_head";
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.scale.x = 0.2;           // 0.2 0.15
    marker.scale.y = 0.2;           // 0.2 0.15
    marker.scale.z = 0.2;           // 0.2 0.15
    marker.pose.position.z = 1.55;  // 1.5 1.0
    marker.id = pids[i];
    marker_array.markers.push_back(marker);

    // Create ID marker and trajectory
    double human_height = 1.85;  // meter
    visualization_msgs::Marker tracking_id;
    //    tracking_id.header.stamp = ros::Time::now();
    tracking_id.header.stamp = now;
    tracking_id.header.frame_id = target_frame;
    tracking_id.ns = "people_id";
    tracking_id.id = pids[i];
    tracking_id.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    tracking_id.pose.position.x = poses[i].position.x;
    tracking_id.pose.position.y = poses[i].position.y;
    tracking_id.pose.position.z = human_height;
    tracking_id.scale.z = 0.4;  // 0.7
    tracking_id.color.a = 1.0;  // 1.0
    tracking_id.color.r = 0.0;  // 1.0
    tracking_id.color.g = 0.0;  // 0.2
    tracking_id.color.b = 0.0;  // 0.0
    tracking_id.text = boost::to_string(pids[i]);
    tracking_id.lifetime = ros::Duration(0.13);
    marker_array.markers.push_back(tracking_id);

    /* for FLOBOT - tracking trajectory */
    if (publish_occluded)
    {
      visualization_msgs::Marker tracking_tr;
      //      tracking_tr.header.stamp = ros::Time::now();
      tracking_tr.header.stamp = now;
      tracking_tr.header.frame_id = target_frame;
      tracking_tr.ns = "people_trajectory";
      tracking_tr.id = pids[i];
      tracking_tr.type = visualization_msgs::Marker::LINE_STRIP;
      geometry_msgs::Point p;
      for (int j = 0; j < previous_poses.size(); j++)
      {
        if (boost::get<0>(previous_poses[j]) == pids[i])
        {
          p.x = boost::get<3>(previous_poses[j]).position.x;
          p.y = boost::get<3>(previous_poses[j]).position.y;
          tracking_tr.points.push_back(p);
        }
      }
      tracking_tr.scale.x = 0.1;
      tracking_tr.color.a = 1.0;
      tracking_tr.color.r = std::max(0.3, (double)(pids[i] % 3) / 3.0);
      tracking_tr.color.g = std::max(0.3, (double)(pids[i] % 6) / 6.0);
      tracking_tr.color.b = std::max(0.3, (double)(pids[i] % 9) / 9.0);
      tracking_tr.lifetime = ros::Duration(1.0);
      marker_array.markers.push_back(tracking_tr);
    }
  }
  pub.publish(marker_array);
}

std::vector<double> PeopleTracker::cartesianToPolar(geometry_msgs::Point point)
{
  ROS_DEBUG("cartesianToPolar: Cartesian point: x: %f, y: %f, z %f", point.x, point.y, point.z);
  std::vector<double> output;
  double dist = sqrt(pow(point.x, 2) + pow(point.y, 2));
  double angle = atan2(point.y, point.x);
  output.push_back(dist);
  output.push_back(angle);
  ROS_DEBUG("cartesianToPolar: Polar point: distance: %f, angle: %f", dist, angle);
  return output;
}

void PeopleTracker::detectorCallback(const geometry_msgs::PoseArray::ConstPtr& pta, std::string detector)
{
  // Publish an empty message to trigger callbacks even when there are no detections.
  // This can be used by nodes which might also want to know when there is no human detected.
  if (pta->poses.size() == 0)
  {
    rc_tracking_msgs::PersonArray tracked_person_array;
    tracked_person_array.header.stamp = pta->header.stamp;
    tracked_person_array.header.frame_id = target_frame;
    pub_tracked_person_array.publish(tracked_person_array);
    return;
  }

  std::vector<geometry_msgs::Point> ppl;
  for (int i = 0; i < pta->poses.size(); i++)
  {
    geometry_msgs::Pose pt = pta->poses[i];

    // Create stamped pose for tf
    geometry_msgs::PoseStamped poseInCamCoords;
    geometry_msgs::PoseStamped poseInTargetCoords;
    poseInCamCoords.header = pta->header;
    poseInCamCoords.pose = pt;

    if (target_frame == poseInCamCoords.header.frame_id)
    {
      poseInTargetCoords = poseInCamCoords;
    }
    else
    {
      // Transform
      try
      {
        // Transform into given traget frame.
        ROS_DEBUG("Transforming received position into %s coordinate system.", target_frame.c_str());
        listener->waitForTransform(poseInCamCoords.header.frame_id, target_frame, poseInCamCoords.header.stamp,
                                   ros::Duration(3.0));
        listener->transformPose(target_frame, ros::Time(0), poseInCamCoords, poseInCamCoords.header.frame_id,
                                poseInTargetCoords);
      }
      catch (tf::TransformException ex)
      {
        ROS_WARN("Failed transform: %s", ex.what());
        return;
      }
    }

    poseInTargetCoords.pose.position.z = 0.0;
    ppl.push_back(poseInTargetCoords.pose.position);
  }
  // match and update tracks with detections
  if (ppl.size())
  {
    if (ekf == NULL)
    {
      if (ukf == NULL)
      {
        pf->addObservation(detector, ppl, pta->header.stamp.toSec(), std::vector<std::string>());
      }
      else
      {
        ukf->addObservation(detector, ppl, pta->header.stamp.toSec(), std::vector<std::string>());
      }
    }
    else
    {
      ekf->addObservation(detector, ppl, pta->header.stamp.toSec(), std::vector<std::string>());
    }
  }

  double time_sec = 0.0;
  std::map<long, std::string> tags;
  try
  {
    std::map<long, std::vector<geometry_msgs::Pose> > ppl;
    if (ekf != NULL)
    {
      ppl = ekf->track(&time_sec, tags, publish_occluded);
    }
    else if (ukf != NULL)
    {
      ppl = ukf->track(&time_sec, tags, publish_occluded);
    }
    else if (pf != NULL)
    {
      ppl = pf->track(&time_sec, tags, publish_occluded);
    }

    if (ppl.size())
    {
      geometry_msgs::Pose closest_person_point;
      std::vector<geometry_msgs::Pose> poses;
      std::vector<geometry_msgs::Pose> vels;
      std::vector<geometry_msgs::Pose> vars;
      std::vector<std::string> uuids;
      std::vector<long> pids;
      std::vector<double> distances;
      std::vector<double> angles;
      double min_dist = DBL_MAX;
      double angle;

      for (std::map<long, std::vector<geometry_msgs::Pose> >::const_iterator it = ppl.begin(); it != ppl.end(); ++it)
      {
        poses.push_back(it->second[0]);
        vels.push_back(it->second[1]);
        vars.push_back(it->second[2]);
        if (tags[it->first] == "")
          uuids.push_back(generateUUID(startup_time_str, it->first));
        else
          uuids.push_back(tags[it->first]);
        pids.push_back(it->first);
      }

      rc_tracking_msgs::PersonArray tracked_person_array;
      tracked_person_array.header.stamp = pta->header.stamp;
      tracked_person_array.header.frame_id = target_frame;
      for (int i = 0; i < poses.size(); i++)
      {
        // Creating and adding Person message
        rc_tracking_msgs::Person tracked_person;
        tracked_person.id = pids[i];
        tracked_person.pose = poses[i];
        tracked_person_array.people.push_back(tracked_person);
      }
      pub_tracked_person_array.publish(tracked_person_array);

      if (pub_marker.getNumSubscribers())
        createVisualisation(pta->header.stamp, poses, pids, pub_marker);

      if (pub_trajectory.getNumSubscribers() && publish_occluded)
        publishTrajectory(poses, vels, vars, pids, pub_trajectory);
    }
    else
    {
      rc_tracking_msgs::PersonArray tracked_person_array;
      tracked_person_array.header.stamp = pta->header.stamp;
      tracked_person_array.header.frame_id = target_frame;
      pub_tracked_person_array.publish(tracked_person_array);
    }
  }
  catch (std::exception& e)
  {
    ROS_ERROR_STREAM("Exception: " << e.what());
  }
  catch (Bayesian_filter::Numeric_exception& e)
  {
    ROS_ERROR_STREAM("Exception: " << e.what());
  }
}

void PeopleTracker::connectCallback(ros::NodeHandle& n)
{
  std::map<std::pair<std::string, std::string>, ros::Subscriber>::const_iterator it;

  for (it = subscribers.begin(); it != subscribers.end(); ++it)
    subscribers[it->first] = n.subscribe<geometry_msgs::PoseArray>(
        it->first.second.c_str(), 1000, boost::bind(&PeopleTracker::detectorCallback, this, _1, it->first.first));
}
}

int main(int argc, char** argv)
{
  // Set up ROS.
  ros::init(argc, argv, "leg_fusion");
  multi_posture_leg_tracker::PeopleTracker* pl = new multi_posture_leg_tracker::PeopleTracker();
  return 0;
}
