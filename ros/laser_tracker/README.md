# Laser Squat Leg Tracker

This work aims to extend the laser tracker by squatting persons. 
Introduction of packages:

- **laser_human_tracker**: laser human tracking module, containing the detector, tracker and occupancy map as well as classifier training nodes. 

---

# 1. Installation

Clone the repository in your workspace and run 

```bash
catkin build
```

Dependencies:

- OpenCV
- [mlpack](https://www.mlpack.org/getstarted.html) 

---

# 2. Demo

## Run trackers

```bash
roslaunch laser_squat_leg_tracker laser_human_tracker.launch
```

**Note**: While robot is moving, set *dynamic_mode* parameter in laser_squat_leg_tracker/config/human_tracker_config.yaml to *true*. The *fixed_frame* should be odometry frame, such as *summit_xl_odom*. 

## Train classifier

Run launch file in laser_squat_leg_tracker/launch/training folder:
   ```bash
   roslaunch laser_squat_leg_tracker train_both_environments.launch # annotate radar cluster samples
   ``` 
Specify *classifier_type* (*opencv_random_forest, opencv_adaboost, mlpack_random_forest, mlpack_adaboost*) and *feature_set_size* (see features defined in laser and radar module for details). You also have to specify the filepath of the training data.After training, the learned model and evaluation results will be saved in the pre-defined path.

## Annotate detection ground truth

1. Set *fixed_frame* in laser detector_config.yaml to corresponding sensor frame (summit_xl_front_laser_link) and run:

   ```bash
   roslaunch laser_squat_leg_tracker annotate_detections_laser.launch # annotate laser cluster samples
   ```

2. Use *Publish Point* tool or *Selected Points Publisher* tool in rviz to choose cluster centers as detection ground truth scan by scan. The ground truth positions are saved in a rosbag file which can be used to train classifiers.

## Annotate track ground truth

1. Set *fixed_frame* in laser detector_config.yaml to laser frame (summit_xl_front_laser_link) and run:

   ```bash
   roslaunch laser_squat_leg_tracker annotate_ground_truth_tracks.launch # annotate gt laser scan by laser scan
   ```

2. Specify the track ID in the AnnotationControl panel in rviz, click the person's position scan by scan. If the person does not appear in the current scan, press *Next* button in the panel or press key *n* on the keyboard to skip to next scan. The ground truth positions of the person are saved in a rosbag file. Afterwards, based on this bag file, run launch file again and annotate another person's position by specifying a new ID. Do this iteratively until all persons have been annotated. 

## Evaluation of tracking performance

1. ```bash
   roslaunch laser_squat_leg_tracker record_estimated_tracks.launch
   ```

   This launch file reads an annotated rosbag, records system outputs and saves them in a new bag.

2. ```bash
   roslaunch laser_squat_leg_tracker calculate_clear_mot.launch
   ```

   This file reads the bag created in the previous step and calculates tracking measures by comparing ground truth and system outputs.

3. To visualize occurrences of tracking errors (misses, false positives and ID switches), run:

   ```bash
   roslaunch laser_squat_leg_tracker view_results.launch
   ```

---

