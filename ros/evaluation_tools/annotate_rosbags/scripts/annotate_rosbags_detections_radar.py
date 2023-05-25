#!/usr/bin/python3

import rospy
import rosbag
from sensor_msgs.msg import LaserScan, PointCloud, Image
from geometry_msgs.msg import PointStamped, PoseArray, Pose
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, Int32, Char
from tf.msg import tfMessage
from rc_tracking_msgs.msg import Person
from visualization_msgs.msg import Marker
import random
import tf
import sys, shutil 
from threading import Lock
import os


class AnnotateRosbagsDetections:
    def __init__(self):  

        self.bag_empty = False

        self.colour = (random.random(), random.random(), random.random())
        self.cur_scan_msg_time = None
        self.next_scan_msg_time = None
        self.annotation_id = int(random.random()*999999)#None
        self.finished_successfully = False
        self.next_scan_msg = None
        self.cur_scan_msg = None
        self.cluster_pose_array = None
        self.mutex = Lock()
        self.prev_time_between_scans = None


        rospy.init_node('annotate_rosbags', anonymous=True)

        # Give Rviz time to boot
        rospy.sleep(3) # not sure this is necessary

        # Read in ROS params
        self.readbag_filename = rospy.get_param("readbag_filename", "error")
        self.savebag_filename = rospy.get_param("savebag_filename", self.readbag_filename)
        self.radar_scan_topic = rospy.get_param("radar_scan_topic", "radar_scan")
        self.laser_scan_topic = rospy.get_param("laser_scan_topic", "laser_scan")
        self.image_topic = rospy.get_param("image_topic", "image")
        self.fixed_frame = rospy.get_param("fixed_frame", "radar")
        self.detection_topic = rospy.get_param("detection_topic", "/leg_cluster_positions")

        # Publisher to view messages from rosbag in Rviz
        self.radar_pub = rospy.Publisher(self.radar_scan_topic, PointCloud, queue_size=10)
        self.laser_pub = rospy.Publisher(self.laser_scan_topic, LaserScan, queue_size=10)
        self.image_pub = rospy.Publisher(self.image_topic, Image, queue_size=10)
        self.tf_pub = rospy.Publisher("tf", tfMessage, queue_size=10)
        self.tf_static_pub = rospy.Publisher("tf_static", tfMessage, queue_size=10)
        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)

        self.listener = tf.TransformListener()        

        # Open the bag file to be annotated
        self.readbag = rosbag.Bag(self.readbag_filename) 
        self.msg_gen = self.readbag.read_messages()

        # Open the bag file to write finished annotations to
        if self.savebag_filename == self.readbag_filename:
            self.savebag = rosbag.Bag(self.savebag_filename + ".temp", 'w')
        else:
            self.savebag = rosbag.Bag(self.savebag_filename, 'w')

        # Subscriber to get annotations. 
        self.key_n_pressed_sub = rospy.Subscriber("key_n_pressed", Char, self.key_n_pressed_callback)
        self.selected_points_sub = rospy.Subscriber("rviz_selected_points", PoseArray, self.selected_points_callback)
        self.map_click_sub = rospy.Subscriber("clicked_point", PointStamped , self.clicked_point_callback)
        self.annnotation_control_sub = rospy.Subscriber("annotation_control", String , self.annnotation_control_callback)

        # To make sure bagfiles are closed properly
        rospy.on_shutdown(self.close_bags)

        # So node doesn't shut down
        rospy.spin()     


    def __del__(self):
        self.close_bags()


    def close_bags(self, not_using_this_var=None):
        self.readbag.close()
        self.savebag.close()

        if self.finished_successfully:
            rospy.loginfo("Finished successfully")
            if self.savebag_filename == self.readbag_filename:
                os.remove(self.readbag_filename)
                os.rename(self.savebag_filename + ".temp", self.savebag_filename)
        else:
            rospy.logwarn("Did NOT finish succesfully! No changes will be saved!")            
            if self.savebag_filename == self.readbag_filename:
                os.remove(self.savebag_filename + ".temp")
            else:
                os.remove(self.savebag_filename)


    def annnotation_control_callback(self, msg):
        if self.next_scan_msg_time is None or self.cur_scan_msg_time is None:
            self.nextScanMsg()
            return

        if msg.data == "Next":
            self.nextScanMsg()
        elif msg.data == "Forwards":
            self.nextScanMsg(10)
        elif msg.data == "End":
            self.nextScanMsg(sys.maxint)
        elif msg.data == "Clear\nprevious":
            rospy.loginfo("option not available in detection annotation")
        elif msg.data == "Next\nmiss":
            rospy.loginfo("option not available in detection annotation")
        elif msg.data == "Next\nid_switch":
            rospy.loginfo("option not available in detection annotation")
        elif msg.data == "Next\nfp":
            rospy.loginfo("option not available in detection annotation")


    def key_n_pressed_callback(self, msg):
        self.nextScanMsg()


    def selected_points_callback(self, selected_points_msg):
        if self.next_scan_msg_time is None or self.cur_scan_msg_time is None:
            self.nextScanMsg()
            return
        if len(selected_points_msg.poses) != 0:
            if self.cluster_pose_array is None:
                self.cluster_pose_array = PoseArray()
                self.cluster_pose_array.header.frame_id = self.fixed_frame
            for cluster in selected_points_msg.poses:
                self.cluster_pose_array.poses.append(cluster)


    def clicked_point_callback(self, clicked_point_msg):
        if self.next_scan_msg_time is None or self.cur_scan_msg_time is None:
            self.nextScanMsg()
            return

        # Save cluster's postion in size of PoseArray used for detector training
        if self.cluster_pose_array is None:
            self.cluster_pose_array = PoseArray()
            self.cluster_pose_array.header.frame_id = self.fixed_frame
        new_cluster_position = Pose()
        new_cluster_position.position.x = clicked_point_msg.point.x
        new_cluster_position.position.y = clicked_point_msg.point.y
        self.cluster_pose_array.poses.append(new_cluster_position)

        # Rviz marker of where annotator clicked to display to rviz temporarily (not saving to rosbag)
        m = Marker()
        m.header.frame_id = self.fixed_frame
        m.id = self.annotation_id
        m.ns = "clicked_points"
        m.type = Marker.SPHERE
        m.lifetime = rospy.Duration(0.05); # Just a short duration to show in rviz
        m.scale.x = 0.2
        m.scale.y = 0.2
        m.scale.z = 0.3
        m.color.r = self.colour[0]
        m.color.g = self.colour[1]
        m.color.b = self.colour[2]
        m.color.a = 1.0
        m.pose.position.x = clicked_point_msg.point.x
        m.pose.position.y = clicked_point_msg.point.y
        m.pose.position.z = 0.1
        self.marker_pub.publish(m)

        self.nextScanMsg()


    # Iterate through the rosbag until we hit a scan message
    # Save that message and break. Play it first time nextScanMsg() is called again.
    # We don't publish and save the scan message found so all the other visualizations markers corresponding to the previous scan 
    # can be displayed and the annotater can annotate based on them.
    # Publish all other messages to Rviz along the way (if display=True)
    def nextScanMsg(self, num_scan_msgs=1, display=True):
        self.mutex.acquire() # TODO necessary?

        if self.cluster_pose_array:
            save_time = (self.next_scan_msg_time - self.cur_scan_msg_time)/2 + self.cur_scan_msg_time
            self.savebag.write(self.detection_topic, self.cluster_pose_array, save_time)
            self.cur_scan_msg.header.frame_id = self.fixed_frame
            self.savebag.write("/training_scan", self.cur_scan_msg, save_time)
            self.cluster_pose_array = None
            self.cur_scan_msg = None

        if self.bag_empty:         
            # Shut down and close rosbags safetly
            self.finished_successfully = True                    
            rospy.loginfo("End of readbag file, shutting down")
            rospy.signal_shutdown("End of readbag file, shutting down")

        # We need <self.prev_time_between_scans> for if we hit the end of the bag and need to extrapolate time
        if self.next_scan_msg_time and self.cur_scan_msg_time:
            self.prev_time_between_scans = self.next_scan_msg_time - self.cur_scan_msg_time


        # First display and save the next_scan_msg
        if self.next_scan_msg:
            topic, msg, time = self.next_scan_msg
            self.cur_scan_msg = msg
            self.cur_scan_msg_time = time                   
            self.next_scan_msg = None
            self.savebag.write(topic, msg, time)
            self.radar_pub.publish(msg)

        # Iterate through rosbag until we've found num_scan_msgs scan messages
        for msg_num in range(0, num_scan_msgs):
            if self.bag_empty:
                break

            topic = None
            while topic !=  self.radar_scan_topic and not self.bag_empty:
                try:
                    # Get the next message from the rosbag
                    topic, msg, time = next(self.msg_gen)

                    if topic == self.radar_scan_topic:
                        if msg_num < num_scan_msgs-1: # We still have some messages to go 
                            # Display and save scan message to rosbag
                            self.savebag.write(topic, msg, time)
                            self.cur_scan_msg_time = time                   
                            if display:
                                self.radar_pub.publish(msg)
                        else:
                            # Last iteration. Save scan message for when nextScanMsg() is called again
                            self.next_scan_msg = (topic, msg, time)
                            self.next_scan_msg_time = time
                    elif topic == self.laser_scan_topic:
#                        msg.header.frame_id = self.fixed_frame
                        self.laser_pub.publish(msg)
                    elif topic == self.image_topic:
                        self.image_pub.publish(msg)
                    elif topic == "/tf":
                        self.tf_pub.publish(msg)
                    elif topic == "/tf_static":
                        self.tf_static_pub.publish(msg)

                except StopIteration:
                    # Passed the last message in the rosbag
                    # Don't close everything yet, but give the annotater one more chance to make an annotation before closing down
                    self.bag_empty = True
                    # Extrapolate time into future
                    if self.prev_time_between_scans is not None:
                        self.next_scan_msg_time = self.prev_time_between_scans + self.cur_scan_msg_time
                    else:
                        rospy.logerr("Error! It appears we didn't read in > 1 scan message. Is radar_scan_topic set correctly?")

        self.mutex.release()





if __name__ == '__main__':
    ar = AnnotateRosbagsDetections()




