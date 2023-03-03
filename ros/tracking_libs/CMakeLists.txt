# in order to add the QT libraries, this option automatically links available QT libraries
# set(CMAKE_AUTOMOC ON)
cmake_minimum_required(VERSION 3.0.2)

# project name
project(rc_tracking_libs)

#forcing the compiler to use c++14 standard
add_definitions(-std=c++14)

## Remove ROS_DEBUG from executable for minimal release builds
if ("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
    add_definitions(-DROSCONSOLE_MIN_SEVERITY=ROSCONSOLE_SEVERITY_INFO)
elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    add_definitions(-DROSCONSOLE_MIN_SEVERITY=ROSCONSOLE_SEVERITY_INFO)
endif()


find_package(OpenCV 4.2.0 REQUIRED)

find_package(Boost REQUIRED system thread math_c99 serialization unit_test_framework)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(CGAL REQUIRED COMPONENTS Core)
set(CGAL_DONT_OVERRIDE_CMAKE_FLAGS TRUE CACHE BOOL "Don't override flags")
message(CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS})
include(${CGAL_USE_FILE})
message(CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS})

if (${Boost_VERSION} LESS 106500)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_MATH_DISABLE_FLOAT128")
endif()
if(NOT Boost_FOUND)
    message(ERROR " Boost not found!")
endif(NOT Boost_FOUND)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  pcl_ros
)

find_package(MLPACK 3.4 REQUIRED)
include_directories(${MLPACK_INCLUDE_DIRS})
message("Mlpack version: ${MLPACK_VERSION}")

## if mlpack has been compiled with openmp
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_CXX_FLAGS "-fopenmp ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(PCL REQUIRED)

catkin_package(
    INCLUDE_DIRS
        include
    LIBRARIES
        kalman_filter_lib
        classifier_lib
    CATKIN_DEPENDS
        roscpp
        pcl_ros
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

### LIBRARIES
add_library(kalman_filter_lib
    src/kalman_filter_lib/hungarian.cpp
    src/kalman_filter_lib/kalman_filter.cpp
    src/kalman_filter_lib/extended_kalman_filter.cpp
    src/kalman_filter_lib/particle_filter.cpp
    src/kalman_filter_lib/unscented_kalman_filter.cpp
    src/kalman_filter_lib/kalman_filter_sh.cpp
)

target_include_directories(kalman_filter_lib
    PUBLIC
        include
        ${catkin_INCLUDE_DIRS}
)

target_link_libraries(kalman_filter_lib
    ${catkin_LIBRARIES}
)

add_library(classifier_lib
    src/classifier_lib/classifier.cpp
    src/classifier_lib/opencv_classifiers/opencv_classifier.cpp
    src/classifier_lib/opencv_classifiers/opencv_random_forest_classifier.cpp
    src/classifier_lib/opencv_classifiers/opencv_adaboost_classifier.cpp
    src/classifier_lib/mlpack_classifiers/mlpack_random_forest_classifier.cpp
    src/classifier_lib/mlpack_classifiers/mlpack_adaboost_classifier.cpp
)

target_include_directories(classifier_lib
    PUBLIC
        include
        ${catkin_INCLUDE_DIRS}
)

target_link_libraries(classifier_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${MLPACK_LIBRARIES}
)

######
###INSTALL

install(TARGETS kalman_filter_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/kalman_filter_lib
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

install(TARGETS classifier_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/classifier_lib
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)
