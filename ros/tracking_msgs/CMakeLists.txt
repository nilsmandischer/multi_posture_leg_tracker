cmake_minimum_required(VERSION 3.0.2)
project(rc_tracking_msgs)


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    message_generation
    geometry_msgs
    std_msgs
)


################################################
############# Declare ROS services #############
################################################

## Generate actions in the 'msg' folder
add_message_files(
    FILES
        Leg.msg
        LegArray.msg
        Person.msg
        PersonArray.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
    DEPENDENCIES
        geometry_msgs
        std_msgs
)


###################################
## catkin specific configuration ##
###################################

catkin_package(
    CATKIN_DEPENDS
        message_runtime
        geometry_msgs
        std_msgs
)

###########
## Build ##
###########

include_directories(
    ${catkin_INCLUDE_DIRS}
)
