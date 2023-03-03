#ifndef RADAR_HUMAN_TRACKER_HUMAN_DETECTOR_H
#define RADAR_HUMAN_TRACKER_HUMAN_DETECTOR_H

#include <set>
#include <string>
#include <vector>
#include <fstream>

#include <ros/ros.h>
#include <ros/publisher.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/PointCloud.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PolygonStamped.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/algorithm.h>
#include <CGAL/assertions.h>

// Custom messages
#include <rc_tracking_msgs/Leg.h>
#include <rc_tracking_msgs/LegArray.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/ml.hpp>

// Classifiers
#include <classifier_lib/classifier.h>
#include <classifier_lib/opencv_classifiers/opencv_random_forest_classifier.h>
#include <classifier_lib/opencv_classifiers/opencv_adaboost_classifier.h>
#include <classifier_lib/mlpack_classifiers/mlpack_random_forest_classifier.h>
#include <classifier_lib/mlpack_classifiers/mlpack_adaboost_classifier.h>

#define FEATURE_SET_0 29   // extended feature set
#define FEATURE_SET_1 30   // extended feature set + distance
#define FEATURE_SET_2 54   // extended feature set + some normalized features
#define FEATURE_SET_3 143  // normalized feature set
#define FEATURE_SET_4 21   // original feature set

namespace multi_posture_leg_tracker{
namespace radar_human_tracker
{
/**
 * @brief A struct representing a single sample (i.e., scan point) from the laser.
 */
class Sample
{
public:
  int index;
  int index_angle;  //!< points on the same beam have the same index_angle
  float intensity;
  float angle;
  float range;
  float x;
  float y;

  /**
   * @brief Return pointer to sample of index <ind>
   */
  static Sample* Extract(int ind, const sensor_msgs::PointCloud& scan);
};

/**
 * @brief The comparator structure allowing the creation of an ordered set of Samples
 */
struct CompareSample
{
  /**
   * @brief The comparator allowing the creation of an ordered set of Samples
   */
  inline bool operator()(const Sample* a, const Sample* b)
  {
    if (std::abs(a->angle - b->angle) < FLT_EPSILON)  // a->angle == b->angle
      return (a->range < b->range);
    else  // a->angle != b->angle
      return (a->angle > b->angle);
  }
};

/**
 * @brief An ordered set of Samples
 *
 * Ordered based on sample index
 */
class SampleSet : public std::set<Sample*, CompareSample>
{
public:
  /**
   * @brief Destructor
   */
  ~SampleSet()
  {
    clear();
  }

  /**
   * @brief Delete all pointers to samples in the set
   */
  void clear();

  /**
   * @brief Get the centroid of the sample points
   * @return Centriod in (x,y,0) (z-element assumed 0)
   */
  tf::Point getPosition();
};

class HumanDetector
{
public:
  /**
   * @brief Default constructor
   */
  HumanDetector(void);

  /**
   * @brief Destructor
   */
  ~HumanDetector(void);

  /**
   * @brief Constructor
   * @param scan Scan to be processed
   */
  HumanDetector(const sensor_msgs::PointCloud& scan);

  /**
   * @brief Remove and delete all references to scan clusters less than a minimum size
   * @param num Minimum number of points in cluster
   */
  void removeLessThan(uint32_t num);

  /**
   * @brief Split scan into clusters
   * @param thresh Euclidian distance threshold for clustering
   */
  void splitConnected(float thresh);

  /**
   * @brief Turns every cluster by 90°, 180°, 270°
   */
  void turnClusters();

  /**
   * @brief Get all the clusters in the scan
   * @return List of clusters
   */
  std::list<SampleSet*>& getClusters()
  {
    return clusters_;
  }

  /**
   * @brief Calculate features for each cluster
   */
  std::vector<float> calcClusterFeatures(const SampleSet* cluster, const sensor_msgs::PointCloud& scan,
                                         int feature_set_size);

  /**
   * @brief Clear the values of the used variables
   * @return void
   */
  void clearAll(void);

  /**
   * @brief Create a sample for every point in the radar point cloud
   * @param radar_scan
   */
  void createSample(const sensor_msgs::PointCloud& radar_scan);

private:
  // ROS system objects
  ros::NodeHandle nh;                    /**< Node Handle reference from embedding node */
  ros::Subscriber filtered_radar_data;   /**< Radar point cloud subscriber */
  ros::Publisher markers_pub_;           /**< Publisher for the markers in rviz */
  ros::Publisher detected_clusters_pub_; /**< Publisher for all detected clusters */

  std::string scan_topic_;                     /**< subscribed radar scan topic */
  std::string detected_clusters_topic_;        /**< detected cluster publishing topic */
  std::string detected_clusters_marker_topic_; /**< detected cluster marker publishing topic */
  std::string fixed_frame_;                    /**< The frame, where clusters are published*/

  std::list<SampleSet*> clusters_; /**< list of clusters*/
  SampleSet* duplicated_samples_;  /**< duplicated radar points used to merge first and last clusters */
  int max_ind_angle_;              /**< total number of beams*/

  std::string classifier_type_;        /**< type of classifier used in detection */
  std::string model_file_;             /**< trained model file*/
  int feature_set_size_;               /**< size of feature set */
  double detection_threshold_;         /**< cluster with confidence greater than threshold can be published */
  double cluster_dist_euclid_;         /**< maximal distance between 2 points in one cluster*/
  int min_points_per_cluster_;         /**< minimum number of points in the cluster*/
  double max_detect_distance_;         /**< maximum distance in the radar set that is processed*/
  int max_detected_clusters_;          /**< number of clusters that have been published*/
  int num_prev_markers_published_;     /**< counter for the markers published*/
  bool use_scan_header_stamp_for_tfs_; /**< Defines which time is used*/
  tf::TransformListener tfl_;          /**< TF listener to get the static tranform*/
  std::shared_ptr<Classifier> classifier_; /**< classifier interface that can point to any specific
                                                           classifier implementation */
  int feat_count_;                                      /**< Number of features taken into account by the classifier */
  bool transform_available;                             /**< availability of tf transform*/
  ros::Time tf_time;                                    /**< transformation time*/
  bool visualize_contour_;                              /**< whether the cluster contour will be visualized in Rviz */
  bool publish_background_; /**< whether clusters classified as backgrounds will be published */

  int scan_num_;          /**< counter of radar scans */
  double execution_time_; /**< total run time until current frame */
  double avg_exec_time_;  /**< average run time */
  double max_exec_time_;  /**< maximum run time */

  /**
   * @brief Callback for every time a filtered point cloud is published
   * @param radar_scan
   */
  void radarCallback(const sensor_msgs::PointCloud::ConstPtr& radar_scan);

  /**
   * @brief Load parameters from parameter server
   */
  void loadParameters(void);

  /**
   * @brief Find out the time that should be used for tfs
   * @param radar_scan [sensor_msgs::PointCloud&] : radar_scan
   */
  void findTransformationTime(const sensor_msgs::PointCloud& radar_scan);

  /**
   * @brief extractContourOfCluster extracts points on the contour of a cluster for feature calculation
   * @param cluster
   * @param front_contour contains points with the shortest range of each beam
   * @param back_contour contains points with the longest range of each beam
   * @param contour is the combination of front contour and back contour
   */
  void extractContourOfCluster(const SampleSet* cluster, std::list<Sample*>& front_contour,
                               std::list<Sample*>& back_contour, std::list<Sample*>& contour);
};

/**
 * @brief Comparison class to order clusters according to their relative distance to the sensor
 */
class CompareClusters
{
public:
  bool operator()(const rc_tracking_msgs::Leg& a, const rc_tracking_msgs::Leg& b);
};

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::FT FT;
typedef K::Point_2 Point;
typedef K::Segment_2 Segment;
typedef CGAL::Alpha_shape_vertex_base_2<K> Vb;
typedef CGAL::Alpha_shape_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds> Triangulation_2;
typedef CGAL::Alpha_shape_2<Triangulation_2> Alpha_shape_2;
typedef Alpha_shape_2::Alpha_shape_edges_iterator Alpha_shape_edges_iterator;

template <class OutputIterator>
void alpha_edges(const Alpha_shape_2& A, OutputIterator out);
}  // namespace radar_human_tracker
} 
#endif  // RADAR_HUMAN_TRACKER_HUMAN_DETECTOR_H
