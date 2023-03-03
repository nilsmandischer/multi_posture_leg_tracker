/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#ifndef LASER_HUMAN_TRACKER_CLUSTER_FEATURES_H
#define LASER_HUMAN_TRACKER_CLUSTER_FEATURES_H

#include "laser_processor.h"
#include <sensor_msgs/LaserScan.h>

#define FEATURE_SET_0 30   // extended feature set
#define FEATURE_SET_1 31   // extended feature set + distance
#define FEATURE_SET_2 53   // extended feature set + some normalized features
#define FEATURE_SET_3 148  // normalized feature set
#define FEATURE_SET_4 17   // original feature set

namespace multi_posture_leg_tracker 
{

namespace laser_human_tracker
{
/**
 * @brief Calculate the geometric features of a cluster of scan points
 */
class ClusterFeatures
{
public:
  /**
   * @brief Calculate the geometric features of a cluster of scan points
   * @param cluster Cluster of interest
   * @param scan Scan containing the cluster
   * @param feature_set_size Chosen size of the feature set
   */
  std::vector<float> calcClusterFeatures(const laser_processor::SampleSet* cluster, const sensor_msgs::LaserScan& scan,
                                         int feature_set_size);
};
}  // namespace laser_human_tracker
}
#endif  // LASER_HUMAN_TRACKER_CLUSTER_FEATURES_H
