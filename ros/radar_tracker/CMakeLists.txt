# in order to add the QT libraries, this option automatically links available QT libraries
cmake_minimum_required(VERSION 3.0.2)

# project name
project(radar_squat_leg_tracker)

#forcing the compiler to use c++14 standard
add_definitions(-std=c++14)

## Remove ROS_DEBUG from executable for minimal release builds
if ("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
    add_definitions(-DROSCONSOLE_MIN_SEVERITY=ROSCONSOLE_SEVERITY_INFO)
elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    add_definitions(-DROSCONSOLE_MIN_SEVERITY=ROSCONSOLE_SEVERITY_INFO)
endif()


find_package(OpenCV 4.2.0 REQUIRED)
message("OpenCV version: ${OpenCV_VERSION}")

find_package(Boost REQUIRED system thread)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS} ${PCL_LIBRARY_DIRS})
add_definitions(${Boost_DEFINITIONS} ${PCL_DEFINITIONS})
#add_definitions(-std=c++11 -Wall -g)
if(NOT Boost_FOUND)
   message(ERROR " Boost not found!")
endif(NOT Boost_FOUND)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(CGAL REQUIRED COMPONENTS Core)
set(CGAL_DONT_OVERRIDE_CMAKE_FLAGS TRUE CACHE BOOL "Don't override flags")
message(CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS})
include(${CGAL_USE_FILE})
message(CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS})

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  rospy
  std_msgs
  geometry_msgs
  visualization_msgs
  tf
  interactive_markers
  rosbag
  image_geometry
  message_generation
  dynamic_reconfigure
  nav_msgs
  pcl_ros
  rc_tracking_libs
  rc_tracking_msgs
#  rviz_visual_tools
)

find_package(Boost REQUIRED COMPONENTS system)

find_package(PCL REQUIRED)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_CXX_FLAGS "-fopenmp ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

###################################
## catkin specific configuration ##
###################################

catkin_package(
    INCLUDE_DIRS
        include
    CATKIN_DEPENDS
        roscpp
        sensor_msgs
        std_msgs
        nav_msgs
        visualization_msgs
        geometry_msgs
        object_msgs
        tf
        pcl_ros
        message_runtime
        dynamic_reconfigure
        rc_tracking_libs
        rc_tracking_msgs
)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

link_directories(
    ${Boost_LIBRARY_DIRS}
    ${PCL_LIBRARY_DIRS}
)

add_definitions(
    ${Boost_DEFINITIONS}
    ${PCL_DEFINITIONS}
)

###########
## Build ##
###########

# do_something node

add_executable(otsu_filter
  src/otsu_filter_node/otsu_filter.cpp
  src/otsu_filter_node.cpp
)
add_dependencies(otsu_filter ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(otsu_filter
  ${catkin_LIBRARIES}
)

add_executable(human_detector_node
  src/human_detector_node/human_detector.cpp
  src/human_detector_node.cpp
)
add_dependencies(human_detector_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(human_detector_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(human_tracker_node
  src/human_tracker_node.cpp
  src/human_tracker_node/human_tracker.cpp
  # src/human_tracker_node/hungarian.cpp
)
add_dependencies(human_tracker_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(human_tracker_node
  ${catkin_LIBRARIES}
)

add_executable(train_classifier
  src/train_classifier_node.cpp
  src/train_classifier_node/train_classifier.cpp
  src/otsu_filter_node/otsu_filter.cpp
  src/human_detector_node/human_detector.cpp
)
add_dependencies(train_classifier ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(train_classifier
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(extract_feature_matrix_node
  src/extract_feature_matrix_node.cpp
  src/extract_feature_matrix_node/extract_feature_matrix.cpp
  src/otsu_filter_node/otsu_filter.cpp
  src/human_detector_node/human_detector.cpp
)
add_dependencies(extract_feature_matrix_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(extract_feature_matrix_node
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

