cmake_minimum_required(VERSION 3.0.2)
project(laser_squat_leg_tracker)

set(CMAKE_CXX_FLAGS "-std=c++14")

find_package(OpenCV 4.2.0 REQUIRED)
message("OpenCV version: ${OpenCV_VERSION}")

find_package(Boost REQUIRED system thread math_c99 serialization unit_test_framework)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
add_definitions(${Boost_DEFINITIONS})
if (${Boost_VERSION} LESS 106500)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_MATH_DISABLE_FLOAT128")
endif()
if(NOT Boost_FOUND)
    message(ERROR " Boost not found!")
endif(NOT Boost_FOUND)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  tf
  interactive_markers
  rosbag
  image_geometry
  rc_tracking_libs
  rc_tracking_msgs
)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_CXX_FLAGS "-fopenmp ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS roscpp rospy std_msgs tf rosbag image_geometry rc_tracking_libs rc_tracking_msgs
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(human_detector_node
  src/human_detector_node.cpp
  src/human_detector_node/human_detector.cpp
  src/laser_processor/laser_processor.cpp
  src/cluster_features/cluster_features.cpp
)
add_dependencies(human_detector_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(human_detector_node
  ${catkin_LIBRARIES}
)

add_executable(train_classifier_node
  src/train_classifier_node.cpp
  src/train_classifier_node/train_classifier.cpp
  src/laser_processor/laser_processor.cpp
  src/cluster_features/cluster_features.cpp
)
add_dependencies(train_classifier_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(train_classifier_node
  ${catkin_LIBRARIES}
)

add_executable(human_tracker_node
  src/human_tracker_node.cpp
  src/human_tracker_node/human_tracker.cpp
)
add_dependencies(human_tracker_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(human_tracker_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(human_tracker_without_map_node
  src/human_tracker_without_map_node.cpp
  src/human_tracker_without_map_node/human_tracker_without_map.cpp
)
add_dependencies(human_tracker_without_map_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(human_tracker_without_map_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)


add_executable(local_occupancy_grid_mapping
  src/local_occupancy_grid_mapping_node.cpp
  src/local_occupancy_grid_mapping_node/local_occupancy_grid_mapping.cpp
  src/laser_processor/laser_processor.cpp
)
add_dependencies(local_occupancy_grid_mapping ${catkin_EXPORTED_TARGETS})
target_link_libraries(local_occupancy_grid_mapping
  ${catkin_LIBRARIES}
)


