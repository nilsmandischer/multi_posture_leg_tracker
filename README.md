# Multi Posture Leg Tracker

## About

The multi posture leg tracker is a software for people tracking with a combination of lidar and radar sensors. It is able to track people walking upright and walking in squats and to distinguish between both postures. It consists of the following packages:


- **laser_tracker**: laser human tracking module, containing the detector, tracker and occupancy map as well as classifier training nodes. 
- **radar_tracker**: radar human tracking module, containing the detector, tracker and as well as classifier training nodes.
- **tracking_libs**: libraries for the tracking algorithms. Contains kalman filter and classifier interface and implementations used by laser and radar detectors. Currently there are four classifier implementations: *opencv_random_forest, opencv_adaboost, mlpack_random_forest, mlpack_adaboost*.
- **leg_fusion**: track fusion module, which takes laser and radar tracks as detection inputs and maintains a number of person tracks. It is based on the bayestracking library.
- **bayestracking**: a multi-sensor multi-target tracking library developed by Bellotto et al. [[10.5281/zenodo.15825](https://doi.org/10.5281/zenodo.15825)]. 
- **tracking_msgs**: detection and tracking messages used by three modules.


## Setup

The following dependencies have to be installed first:
- CGAL 5.3
- mlpack 3.4.2
- openCV 4.2.0
- Boost
- Eigen

We used the tracker with these software versions, but it is possible that other versions work as well.

Clone the repository into your catkin_ws.

Setup the needed ros packages:

```
rosdep install --from-paths src --ignore-src -r -y
```

Build the workspace:

```
catkin build
```

## Usage

The 3 main components of the system are the following:
- laser_tracker
- radar_tracker
- leg_fusion

For training of the models and correct parameter settings, each of these modules has its own readme. 
For the Laser and Radar leg tracker, we provide already trained models.

Start all components alltogether with

```
roslaunch leg_fusion startup_all.launch
```

## License

This project is licensed under the MIT License.

## Authors

This software was written at the Institute of Mechanism Theory, Machine Dynamics and Robotics at Rheinisch-Westfälische Technische Hochschule Aachen.

