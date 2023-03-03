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

#include "laser_squat_leg_tracker/cluster_features.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>

#include <algorithm>
#include <limits>

namespace multi_posture_leg_tracker 
{

namespace laser_human_tracker
{
std::vector<float> ClusterFeatures::calcClusterFeatures(const laser_processor::SampleSet* cluster,
                                                        const sensor_msgs::LaserScan& scan, int feature_set_size)
{
  /**
   * Feature: Number of points
   */
  int num_points = cluster->size();

  // Compute mean and median points for future use
  float x_mean = 0.0;
  float y_mean = 0.0;
  std::vector<float> x_median_set;
  std::vector<float> y_median_set;
  for (laser_processor::SampleSet::iterator i = cluster->begin(); i != cluster->end(); i++)
  {
    x_mean += ((*i)->x) / num_points;
    y_mean += ((*i)->y) / num_points;
    x_median_set.push_back((*i)->x);
    y_median_set.push_back((*i)->y);
  }

  sort(x_median_set.begin(), x_median_set.end());
  sort(y_median_set.begin(), y_median_set.end());

  float x_median = 0.5 * (*(x_median_set.begin() + (num_points - 1) / 2) + *(x_median_set.begin() + num_points / 2));
  float y_median = 0.5 * (*(y_median_set.begin() + (num_points - 1) / 2) + *(y_median_set.begin() + num_points / 2));

  /**
   * Feature: std from mean point
   * Feature: mean average deviation from median point
   */
  double sum_std_diff = 0.0;
  double sum_med_diff = 0.0;
  double std_x = 0.0;
  double std_y = 0.0;
  // used for computing kurtosis
  double sum_kurt = 0.0;

  for (laser_processor::SampleSet::iterator i = cluster->begin(); i != cluster->end(); i++)
  {
    double diff_x = pow((*i)->x - x_mean, 2);
    double diff_y = pow((*i)->y - y_mean, 2);
    std_x += diff_x;
    std_y += diff_y;
    sum_kurt += pow(diff_x + diff_y, 2);  // sum((Y - Y_bar)^4)

    sum_med_diff += sqrt(pow((*i)->x - x_median, 2) + pow((*i)->y - y_median, 2));
  }
  sum_std_diff = std_x + std_y;
  float std = sqrt(1.0 / (num_points - 1.0) * sum_std_diff);
  float avg_median_dev = sum_med_diff / num_points;

  // used for computing aspect ratio
  std_x = sqrt(std_x / (num_points - 1.0));
  std_y = sqrt(std_y / (num_points - 1.0));

  // Get first and last points in cluster
  laser_processor::SampleSet::iterator first = cluster->begin();
  laser_processor::SampleSet::iterator last = cluster->end();
  --last;

  /**
   * Feature: width
   * euclidian distance between first + last points
   * min_dist: min distance between points
   * max_dist: max distance between points
   * ratio_min_max_dist: ratio between min and max distance between points
   */
  float width = sqrt(pow((*first)->x - (*last)->x, 2) + pow((*first)->y - (*last)->y, 2));
  float min_dist = std::numeric_limits<float>::infinity();
  float max_dist = 0.0;
  float ratio_min_max_dist = 0.0;

  /**
   * Feature: linearity
   * min_lin_err: min error of points to fitted line
   * max_lin_err: max error of points to fitted line
   * ratio_min_max_lin_err
   */
  CvMat* points = cvCreateMat(num_points, 2, CV_64FC1);
  {
    int j = 0;
    for (laser_processor::SampleSet::iterator i = cluster->begin(); i != cluster->end(); i++)
    {
      cvmSet(points, j, 0, (*i)->x - x_mean);
      cvmSet(points, j, 1, (*i)->y - y_mean);
      j++;
    }
  }

  CvMat* W = cvCreateMat(2, 2, CV_64FC1);
  CvMat* U = cvCreateMat(num_points, 2, CV_64FC1);
  CvMat* V = cvCreateMat(2, 2, CV_64FC1);
  cvSVD(points, W, U, V);

  CvMat* rot_points = cvCreateMat(num_points, 2, CV_64FC1);
  cvMatMul(U, W, rot_points);

  float linearity = 0.0;
  float min_lin_err = std::numeric_limits<float>::infinity();
  float max_lin_err = 0.0;
  float ratio_min_max_lin_err = 0.0;
  for (int i = 0; i < num_points; i++)
  {
    float diff = cvmGet(rot_points, i, 1);
    if (diff < min_lin_err)
      min_lin_err = diff;
    if (diff > max_lin_err)
      max_lin_err = diff;
    linearity += pow(cvmGet(rot_points, i, 1), 2);
  }
  if (max_lin_err != 0.0)
    ratio_min_max_lin_err = min_lin_err / max_lin_err;

  cvReleaseMat(&points);
  points = 0;
  cvReleaseMat(&W);
  W = 0;
  cvReleaseMat(&U);
  U = 0;
  cvReleaseMat(&V);
  V = 0;
  cvReleaseMat(&rot_points);
  rot_points = 0;

  /**
   * Feature: circularity
   * min_dist2center: min difference from points to center of fitted circle
   * max_dist2center: max difference from points to center of fitted circle
   * ratio_min_max_dist2center
   */
  CvMat* A = cvCreateMat(num_points, 3, CV_64FC1);
  CvMat* B = cvCreateMat(num_points, 1, CV_64FC1);
  {
    int j = 0;
    for (laser_processor::SampleSet::iterator i = cluster->begin(); i != cluster->end(); i++)
    {
      float x = (*i)->x;
      float y = (*i)->y;

      cvmSet(A, j, 0, -2.0 * x);
      cvmSet(A, j, 1, -2.0 * y);
      cvmSet(A, j, 2, 1);

      cvmSet(B, j, 0, -pow(x, 2) - pow(y, 2));
      j++;
    }
  }
  CvMat* sol = cvCreateMat(3, 1, CV_64FC1);

  cvSolve(A, B, sol, CV_SVD);

  float xc = cvmGet(sol, 0, 0);
  float yc = cvmGet(sol, 1, 0);
  float rc = sqrt(pow(xc, 2) + pow(yc, 2) - cvmGet(sol, 2, 0));

  cvReleaseMat(&A);
  A = 0;
  cvReleaseMat(&B);
  B = 0;
  cvReleaseMat(&sol);
  sol = 0;

  float circularity = 0.0;
  float min_dist2center = std::numeric_limits<float>::infinity();
  float max_dist2center = 0.0;
  float ratio_min_max_dist2center = 0.0;

  for (laser_processor::SampleSet::iterator i = cluster->begin(); i != cluster->end(); i++)
  {
    float dist = sqrt(pow(xc - (*i)->x, 2) + pow(yc - (*i)->y, 2));
    if (dist < min_dist2center)
      min_dist2center = dist;
    if (dist > max_dist2center)
      max_dist2center = dist;
    circularity += pow(rc - dist, 2);
  }

  if (max_dist2center != 0.0)
    ratio_min_max_dist2center = min_dist2center / max_dist2center;

  /**
   * Feature: radius
   */
  float radius = rc;

  /**
   * Feature: boundary length
   * Feature: boundary regularity
   */
  float boundary_length = 0.0;
  float last_boundary_seg = 0.0;

  float boundary_regularity = 0.0;
  double sum_boundary_reg_sq = 0.0;

  /**
   * Feature: mean curvature
   */
  float mean_curvature = 0.0;

  laser_processor::SampleSet::iterator left = cluster->begin();
  left++;
  left++;
  laser_processor::SampleSet::iterator mid = cluster->begin();
  mid++;
  laser_processor::SampleSet::iterator right = cluster->begin();

  /**
   * Feature: mean angular difference
   */
  float ang_diff = 0.0;

  while (left != cluster->end())
  {
    float mlx = (*left)->x - (*mid)->x;
    float mly = (*left)->y - (*mid)->y;
    float L_ml = sqrt(mlx * mlx + mly * mly);

    float mrx = (*right)->x - (*mid)->x;
    float mry = (*right)->y - (*mid)->y;
    float L_mr = sqrt(mrx * mrx + mry * mry);

    float lrx = (*left)->x - (*right)->x;
    float lry = (*left)->y - (*right)->y;
    float L_lr = sqrt(lrx * lrx + lry * lry);

    if (L_mr < min_dist)
      min_dist = L_mr;
    if (L_mr > max_dist)
      max_dist = L_mr;

    boundary_length += L_mr;
    sum_boundary_reg_sq += L_mr * L_mr;
    last_boundary_seg = L_ml;

    float A = (mlx * mrx + mly * mry) / pow(L_mr, 2);
    float B = (mlx * mry - mly * mrx) / pow(L_mr, 2);

    float th = atan2(B, A);

    if (th < 0)
      th += 2 * M_PI;

    ang_diff += th / num_points;

    float s = 0.5 * (L_ml + L_mr + L_lr);
    float area_square = std::max(s * (s - L_ml) * (s - L_mr) * (s - L_lr), static_cast<float>(0));
    float area = sqrt(area_square);

    if (th > 0)
      mean_curvature += 4 * (area) / (L_ml * L_mr * L_lr * num_points);
    else
      mean_curvature -= 4 * (area) / (L_ml * L_mr * L_lr * num_points);

    left++;
    mid++;
    right++;
  }

  if (last_boundary_seg < min_dist)
    min_dist = last_boundary_seg;
  if (last_boundary_seg > max_dist)
    max_dist = last_boundary_seg;
  if (max_dist != 0.0)
    ratio_min_max_dist = min_dist / max_dist;

  boundary_length += last_boundary_seg;
  sum_boundary_reg_sq += last_boundary_seg * last_boundary_seg;

  boundary_regularity = sqrt((sum_boundary_reg_sq - pow(boundary_length, 2) / num_points) / (num_points - 1));

  /**
   * Feature: incribed angle variance
   * Feature: std incribed angle variance
   */
  first = cluster->begin();
  mid = cluster->begin();
  mid++;
  last = cluster->end();
  last--;

  double sum_iav = 0.0;
  double sum_iav_sq = 0.0;

  while (mid != last)
  {
    float mlx = (*first)->x - (*mid)->x;
    float mly = (*first)->y - (*mid)->y;

    float mrx = (*last)->x - (*mid)->x;
    float mry = (*last)->y - (*mid)->y;
    float L_mr = sqrt(mrx * mrx + mry * mry);

    float A = (mlx * mrx + mly * mry) / pow(L_mr, 2);
    float B = (mlx * mry - mly * mrx) / pow(L_mr, 2);

    float th = atan2(B, A);

    if (th < 0)
      th += 2 * M_PI;

    sum_iav += th;
    sum_iav_sq += th * th;

    mid++;
  }

  float iav = sum_iav / num_points;
  float std_iav = sqrt((sum_iav_sq - pow(sum_iav, 2) / num_points) / (num_points - 1));

  /**
   * Feature: aspect ratio (1D)
   */
  float aspect_ratio = (1.0 + MIN(std_x, std_y)) / (1.0 + MAX(std_x, std_y));

  /**
   * Feature: kurtosis (1D)
   */
  float num = sum_kurt / num_points;
  float std_n = sqrt(1.0 / num_points * sum_std_diff);
  float denom = pow(std_n, 4);
  float kurtosis = num / denom;

  /**
   * Feature: Occluded right
   * Feature: Occluded left
   * Feature: Jump distance from preceeding segment
   * Feature: Jump distance from succeeding segment
   */
  int prev_ind = (*first)->index - 1;
  int next_ind = (*last)->index + 1;

  float prev_jump = 0;
  float next_jump = 0;

  float occluded_left = 1;
  float occluded_right = 1;

  if (prev_ind >= 0)
  {
    laser_processor::Sample* prev = laser_processor::Sample::Extract(prev_ind, scan);
    if (prev != NULL)
    {
      prev_jump = sqrt(pow((*first)->x - prev->x, 2) + pow((*first)->y - prev->y, 2));

      if ((*first)->range < prev->range or prev->range < 0.01)
        occluded_left = 0;

      delete prev;
    }
  }

  if (next_ind < (int)scan.ranges.size())
  {
    laser_processor::Sample* next = laser_processor::Sample::Extract(next_ind, scan);
    if (next != NULL)
    {
      next_jump = sqrt(pow((*last)->x - next->x, 2) + pow((*last)->y - next->y, 2));

      if ((*last)->range < next->range or next->range < 0.01)
        occluded_right = 0;

      delete next;
    }
  }

  /**
   * Feature: polygon area
   */
  double sum_area = 0.0;
  right = cluster->begin();
  mid = cluster->begin();
  mid++;
  while (mid != cluster->end())
  {
    sum_area += ((*right)->x * (*mid)->y) - ((*mid)->x * (*right)->y);
    right++;
    mid++;
  }
  // close polygon
  sum_area += ((*last)->x * (*first)->y) - ((*first)->x * (*last)->y);
  sum_area *= 0.5;
  float polygon_area = fabs(sum_area);

  /**
   * Feature: distance to laser scanner
   */
  float distance = sqrt(x_median * x_median + y_median * y_median);

  /**
   * Feature: number of local minima
   */
  size_t num_local_minima[15];
  float noise[15];
  bool in_minima[15];
  for (size_t i = 0; i < 15; i++)
  {
    num_local_minima[i] = 0;
    noise[i] = 0.01 * i;
    in_minima[i] = false;
  }
  float min_range = FLT_MAX;
  float max_range = -1.0;
  float last_range = -1.0;
  for (laser_processor::SampleSet::iterator i = cluster->begin(); i != cluster->end(); i++)
  {
    const float range = (*i)->range;
    min_range = std::min(min_range, range);
    max_range = std::max(max_range, range);
    for (size_t j = 0; j < 15; j++)
    {
      if (range < last_range - noise[j])
      {
        if (!in_minima[j])
        {
          num_local_minima[j]++;
          in_minima[j] = true;
        }
      }
      else
      {
        in_minima[j] = false;
      }
    }
    last_range = range;
  }
  float diff_max_min_range = max_range - min_range;
  float ratio_min_max_range = min_range / max_range;

  // Add features
  std::vector<float> features;

  switch (feature_set_size)
  {
    case 0:  // 30 (extended feature set)
      // features from "Using Boosted Features for the Detection of People in 2D Range Data"
      features.push_back(num_points);      // 1
      features.push_back(std);             // 2
      features.push_back(avg_median_dev);  // 3

      features.push_back(width);               // 4
      features.push_back(min_dist);            // 5
      features.push_back(max_dist);            // 6
      features.push_back(ratio_min_max_dist);  // 7

      features.push_back(linearity);              // 8
      features.push_back(min_lin_err);            // 9
      features.push_back(max_lin_err);            // 10
      features.push_back(ratio_min_max_lin_err);  // 11

      features.push_back(circularity);                // 12
      features.push_back(radius);                     // 13
      features.push_back(min_dist2center);            // 14
      features.push_back(max_dist2center);            // 15
      features.push_back(ratio_min_max_dist2center);  // 16

      features.push_back(boundary_length);  // 17

      features.push_back(boundary_regularity);  // 18

      features.push_back(mean_curvature);  // 19

      features.push_back(ang_diff);  // 20
      // feature from paper which cannot be calculated here: mean speed

      // Inscribed angular variance, I believe. Not sure what paper this is from
      features.push_back(iav);      // 21
      features.push_back(std_iav);  // 22

      // new features from L. Spinello
      features.push_back(aspect_ratio);  // 23

      features.push_back(kurtosis);  // 24

      features.push_back(polygon_area);  // 25

      // features from Angus
      features.push_back(occluded_right);  // 26
      features.push_back(occluded_left);   // 27

      features.push_back(num_local_minima[0]);  // 28
      features.push_back(diff_max_min_range);   // 29
      features.push_back(ratio_min_max_range);  // 30
      break;

    case 1:                                // 31 (extended feature set + distance)
      features.push_back(num_points);      // 1
      features.push_back(std);             // 2
      features.push_back(avg_median_dev);  // 3

      features.push_back(width);               // 4
      features.push_back(min_dist);            // 5
      features.push_back(max_dist);            // 6
      features.push_back(ratio_min_max_dist);  // 7

      features.push_back(linearity);              // 8
      features.push_back(min_lin_err);            // 9
      features.push_back(max_lin_err);            // 10
      features.push_back(ratio_min_max_lin_err);  // 11

      features.push_back(circularity);                // 12
      features.push_back(radius);                     // 13
      features.push_back(min_dist2center);            // 14
      features.push_back(max_dist2center);            // 15
      features.push_back(ratio_min_max_dist2center);  // 16

      features.push_back(boundary_length);  // 17

      features.push_back(boundary_regularity);  // 18

      features.push_back(mean_curvature);  // 19

      features.push_back(ang_diff);  // 20

      features.push_back(iav);      // 21
      features.push_back(std_iav);  // 22

      features.push_back(aspect_ratio);  // 23

      features.push_back(kurtosis);  // 24

      features.push_back(polygon_area);  // 25

      features.push_back(occluded_right);  // 26
      features.push_back(occluded_left);   // 27

      features.push_back(num_local_minima[0]);  // 28
      features.push_back(diff_max_min_range);   // 29
      features.push_back(ratio_min_max_range);  // 30
      features.push_back(distance);             // 31
      break;

    case 2:                                           // 53 (extended feature set + some normalized features)
      features.push_back(num_points);                 // 1
      features.push_back(std);                        // 2
      features.push_back(avg_median_dev);             // 3
      features.push_back(width);                      // 4
      features.push_back(min_dist);                   // 5
      features.push_back(max_dist);                   // 6
      features.push_back(ratio_min_max_dist);         // 7
      features.push_back(linearity);                  // 8
      features.push_back(min_lin_err);                // 9
      features.push_back(max_lin_err);                // 10
      features.push_back(ratio_min_max_lin_err);      // 11
      features.push_back(circularity);                // 12
      features.push_back(radius);                     // 13
      features.push_back(min_dist2center);            // 14
      features.push_back(max_dist2center);            // 15
      features.push_back(ratio_min_max_dist2center);  // 16
      features.push_back(boundary_length);            // 17
      features.push_back(boundary_regularity);        // 18
      features.push_back(mean_curvature);             // 19
      features.push_back(ang_diff);                   // 20
      features.push_back(iav);                        // 21
      features.push_back(std_iav);                    // 22
      features.push_back(aspect_ratio);               // 23
      features.push_back(kurtosis);                   // 24
      features.push_back(polygon_area);               // 25
      features.push_back(occluded_right);             // 26
      features.push_back(occluded_left);              // 27
      features.push_back(num_local_minima[0]);        // 28
      features.push_back(diff_max_min_range);         // 29
      features.push_back(ratio_min_max_range);        // 30

      features.push_back(num_points * distance);           // 1
      features.push_back(min_dist / distance);             // 5
      features.push_back(ratio_min_max_dist / distance);   // 7
      features.push_back(mean_curvature * distance);       // 19
      features.push_back(ang_diff * distance);             // 20
      features.push_back(std_iav / distance);              // 22
      features.push_back(ratio_min_max_range / distance);  // 30

      features.push_back(std / num_points);                        // 2
      features.push_back(avg_median_dev / num_points);             // 3
      features.push_back(width / num_points);                      // 4
      features.push_back(min_dist * num_points);                   // 5
      features.push_back(ratio_min_max_dist * num_points);         // 7
      features.push_back(linearity / num_points);                  // 8
      features.push_back(min_lin_err * num_points);                // 9
      features.push_back(max_lin_err / num_points);                // 10
      features.push_back(circularity / num_points);                // 12
      features.push_back(ratio_min_max_dist2center * num_points);  // 16
      features.push_back(boundary_length / num_points);            // 17
      features.push_back(mean_curvature / num_points);             // 19
      features.push_back(ang_diff / num_points);                   // 20
      features.push_back(iav / num_points);                        // 21
      features.push_back(std_iav * num_points);                    // 22
      features.push_back(ratio_min_max_range * num_points);        // 30
      break;

    case 3:                                // 148 (normalized feature set)
      features.push_back(num_points);      // 1
      features.push_back(std);             // 2
      features.push_back(avg_median_dev);  // 3

      features.push_back(width);               // 4
      features.push_back(min_dist);            // 5
      features.push_back(max_dist);            // 6
      features.push_back(ratio_min_max_dist);  // 7

      features.push_back(linearity);              // 8
      features.push_back(min_lin_err);            // 9
      features.push_back(max_lin_err);            // 10
      features.push_back(ratio_min_max_lin_err);  // 11

      features.push_back(circularity);                // 12
      features.push_back(radius);                     // 13
      features.push_back(min_dist2center);            // 14
      features.push_back(max_dist2center);            // 15
      features.push_back(ratio_min_max_dist2center);  // 16

      features.push_back(boundary_length);  // 17

      features.push_back(boundary_regularity);  // 18

      features.push_back(mean_curvature);  // 19

      features.push_back(ang_diff);  // 20

      features.push_back(iav);      // 21
      features.push_back(std_iav);  // 22

      features.push_back(aspect_ratio);  // 23

      features.push_back(kurtosis);  // 24

      features.push_back(polygon_area);  // 25

      features.push_back(occluded_right);  // 26
      features.push_back(occluded_left);   // 27

      features.push_back(num_local_minima[0]);  // 28
      features.push_back(diff_max_min_range);   // 29
      features.push_back(ratio_min_max_range);  // 30

      for (uint i = 0; i < FEATURE_SET_0; i++)
      {
        features.push_back(features[i] / distance);
      }
      for (uint i = 0; i < FEATURE_SET_0; i++)
      {
        features.push_back(features[i] * distance);
      }

      for (uint i = 1; i < FEATURE_SET_0; i++)
      {
        features.push_back(features[i] / num_points);
      }
      for (uint i = 1; i < FEATURE_SET_0; i++)
      {
        features.push_back(features[i] * num_points);
      }
      break;

    case 4:  // 17 (original feature set)
      features.push_back(num_points);
      features.push_back(std);
      features.push_back(avg_median_dev);
      features.push_back(width);
      features.push_back(linearity);
      features.push_back(circularity);
      features.push_back(radius);
      features.push_back(boundary_length);
      features.push_back(boundary_regularity);
      features.push_back(mean_curvature);
      features.push_back(ang_diff);
      features.push_back(iav);
      features.push_back(std_iav);
      // New features from Angus
      features.push_back(distance);
      features.push_back(distance / num_points);
      features.push_back(occluded_right);
      features.push_back(occluded_left);
      break;
  }

  return features;
}
}  // namespace laser_human_tracker
}