#include "rviz/selection/selection_manager.h"
#include "rviz/viewport_mouse_event.h"
#include "rviz/display_context.h"
#include "rviz/selection/forwards.h"
#include "rviz/properties/property_tree_model.h"
#include "rviz/properties/property.h"
#include "rviz/properties/vector_property.h"
#include "rviz/view_manager.h"
#include "rviz/view_controller.h"
#include "OGRE/OgreCamera.h"

#include "selected_points_publisher/SelectedPointsPublisher.h"

#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/PointCloud2.h>
#include <QVariant>

#include <visualization_msgs/Marker.h>

namespace rviz_plugin_selected_points_publisher
{
SelectedPointsPublisher::SelectedPointsPublisher()
{
  updateTopic();
}

SelectedPointsPublisher::~SelectedPointsPublisher()
{
}

void SelectedPointsPublisher::updateTopic()
{
  rviz_cloud_topic_ = std::string("/rviz_selected_points");

  rviz_selected_pub_ = nh_.advertise<geometry_msgs::PoseArray>(rviz_cloud_topic_.c_str(), 1);
  key_n_pressed_pub_ = nh_.advertise<std_msgs::Char>("/key_n_pressed", 10);
  key_c_pressed_pub_ = nh_.advertise<std_msgs::Char>("/key_c_pressed", 10);

  selected_points_.reset(new geometry_msgs::PoseArray());

  num_selected_points_ = 0;
}

int SelectedPointsPublisher::processKeyEvent(QKeyEvent* event, rviz::RenderPanel* panel)
{
  if (event->type() == QKeyEvent::KeyPress)
  {
    if (event->key() == 'n' || event->key() == 'N')
    {
      std_msgs::Char n;
      n.data = 'n';
      key_n_pressed_pub_.publish(n);
      rviz::SelectionManager* sel_manager = context_->getSelectionManager();
      rviz::M_Picked selection = sel_manager->getSelection();
      sel_manager->removeSelection(selection);
      selected_points_.reset(new geometry_msgs::PoseArray());
      num_selected_points_ = 0;
    }
    else if (event->key() == 'c' || event->key() == 'C')
    {
      std_msgs::Char c;
      c.data = 'c';
      key_c_pressed_pub_.publish(c);
      rviz::SelectionManager* sel_manager = context_->getSelectionManager();
      rviz::M_Picked selection = sel_manager->getSelection();
      sel_manager->removeSelection(selection);
      selected_points_.reset(new geometry_msgs::PoseArray());
      num_selected_points_ = 0;
    }
  }
}

int SelectedPointsPublisher::processMouseEvent(rviz::ViewportMouseEvent& event)
{
  int flags = rviz::SelectionTool::processMouseEvent(event);

  // determine current selection mode
  if (event.alt())
  {
    selecting_ = false;
  }
  else
  {
    if (event.leftDown())
    {
      selecting_ = true;
    }
  }

  if (selecting_)
  {
    if (event.leftUp())
    {
      ROS_INFO_STREAM_NAMED("SelectedPointsPublisher.processKeyEvent", "Using selected area to find a new bounding box "
                                                                       "and publish the points inside of it");
      this->processSelectedAreaAndFindPoints();
    }
  }
  return flags;
}

int SelectedPointsPublisher::processSelectedAreaAndFindPoints()
{
  rviz::SelectionManager* sel_manager = context_->getSelectionManager();
  rviz::M_Picked selection = sel_manager->getSelection();
  rviz::PropertyTreeModel* model = sel_manager->getPropertyModel();

  // Generate a ros point cloud message with the selected points in rviz
  selected_points_.reset(new geometry_msgs::PoseArray());
  selected_points_->header.frame_id = context_->getFixedFrame().toStdString();

  int i = 0;
  while (model->hasIndex(i, 0))
  {
    QModelIndex child_index = model->index(i, 0);

    rviz::Property* child = model->getProp(child_index);
    QString string_marker("Marker");  // Only select and publish markers
    if (child->getName().contains(string_marker))
    {
      rviz::VectorProperty* subchild = (rviz::VectorProperty*)child->childAt(0);
      Ogre::Vector3 vec = subchild->getVector();

      geometry_msgs::Pose point;
      point.position.x = vec.x;
      point.position.y = vec.y;
      selected_points_->poses.push_back(point);
    }

    i++;
  }

  num_selected_points_ = i;
  ROS_INFO_STREAM_NAMED("SelectedPointsPublisher._processSelectedAreaAndFindPoints",
                        "Number of points in the selected area: " << num_selected_points_);

  selected_points_->header.stamp = ros::Time::now();

  if (!selected_points_->poses.empty())
  {
    rviz_selected_pub_.publish(*selected_points_);
  }

  return 0;
}

}  // namespace rviz_plugin_selected_points_publisher

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(rviz_plugin_selected_points_publisher::SelectedPointsPublisher, rviz::Tool)
