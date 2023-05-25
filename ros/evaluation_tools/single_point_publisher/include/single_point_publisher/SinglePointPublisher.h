#ifndef SINGLE_POINT_PUBLISHER_H
#define SINGLE_POINT_PUBLISHER_H

#ifndef Q_MOC_RUN  // See: https://bugreports.qt-project.org/browse/QTBUG-22829
#include <ros/node_handle.h>
#include <ros/publisher.h>

#include "rviz/tool.h"

#include <QCursor>
#include <QObject>
#endif

#include "rviz/default_plugin/tools/point_tool.h"
#include <std_msgs/Char.h>
#include <std_msgs/Int32.h>

namespace rviz_plugin_single_point_publisher
{
class SinglePointPublisher;

class SinglePointPublisher : public rviz::PointTool
{
  Q_OBJECT
public:
  SinglePointPublisher();
  virtual ~SinglePointPublisher();

  virtual int processKeyEvent(QKeyEvent* event, rviz::RenderPanel* panel);

public Q_SLOTS:
  /*
   * Creates the ROS topic
   */
  void updateTopic();

protected:
  int processSelectedAreaAndFindPoints();
  ros::NodeHandle nh_;
  ros::Publisher key_n_pressed_pub_;
  ros::Publisher id_pub_;
};
}  // namespace rviz_plugin_single_point_publisher

#endif  // SINGLE_POINT_PUBLISHER_H
