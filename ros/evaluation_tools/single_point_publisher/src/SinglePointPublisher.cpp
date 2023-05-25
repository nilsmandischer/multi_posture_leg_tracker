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

#include "single_point_publisher/SinglePointPublisher.h"

#include <ros/ros.h>
#include <ros/time.h>
#include <QVariant>

namespace rviz_plugin_single_point_publisher
{
SinglePointPublisher::SinglePointPublisher()
{
  updateTopic();
}

SinglePointPublisher::~SinglePointPublisher()
{
}

void SinglePointPublisher::updateTopic()
{
  id_pub_ = nh_.advertise<std_msgs::Int32>("/id", 10);
  key_n_pressed_pub_ = nh_.advertise<std_msgs::Char>("/key_n_pressed", 10);
}

int SinglePointPublisher::processKeyEvent(QKeyEvent* event, rviz::RenderPanel* panel)
{
  if (event->type() == QKeyEvent::KeyPress)
  {
    if (event->key() >= '0' && event->key() <= '9')
    {
      std_msgs::Int32 id;
      id.data = event->key() - '0';
      id_pub_.publish(id);
    }
    else if (event->key() == 'n' || event->key() == 'N')
    {
      std_msgs::Char n;
      n.data = event->key();
      key_n_pressed_pub_.publish(n);
    }
  }
}

}  // namespace rviz_plugin_single_point_publisher

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(rviz_plugin_single_point_publisher::SinglePointPublisher, rviz::Tool)
