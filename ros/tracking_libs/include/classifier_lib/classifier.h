#ifndef CLASSIFIERS_CLASSIFIER_H
#define CLASSIFIERS_CLASSIFIER_H

#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>

namespace multi_posture_leg_tracker {

/**
 * @brief The Classifier class
 * Abstract classifier interface that any concrete implementations need to inherit so that laser or radar detector can
 * directly utilize them.
 */
class Classifier
{
public:
  Classifier(ros::NodeHandle& nh);

  /**
   * @brief classifyFeatureVector Assign one single cluster label
   * @param feature_vector Features extracted from a cluster
   * @param label {leg, squatting person, non-human}
   * @param confidence Confidence level to be human
   */
  virtual void classifyFeatureVector(const std::vector<float>& feature_vector, float& label, float& confidence) = 0;

  /**
   * @brief train Train model based on features extracted from a number of cluster samples
   * @param leg_feature_matrix Features extracted from leg samples
   * @param squat_feature_matrix Features extracted from squatting samples
   * @param neg_feature_matrix Features extracted from non-human samples
   * @param eval_filename_path File path to save the evaluation results
   */
  virtual void train(std::vector<std::vector<float>>& leg_feature_matrix,
                     std::vector<std::vector<float>>& squat_feature_matrix,
                     std::vector<std::vector<float>>& neg_feature_matrix, const std::string& eval_filename_path) = 0;

  /**
   * @brief test Test trained model
   * @param leg_feature_matrix
   * @param squat_feature_matrix
   * @param neg_feature_matrix
   * @param eval_filename File path to save the test results
   */
  virtual void test(std::vector<std::vector<float>>& leg_feature_matrix,
                    std::vector<std::vector<float>>& squat_feature_matrix,
                    std::vector<std::vector<float>>& neg_feature_matrix, const std::string& eval_filename);

  virtual bool saveModel(const std::string& filename) = 0;

  virtual bool loadModel(const std::string& filename) = 0;

  /**
   * @brief getFeatureSetSize
   * @return Dimension of the feature vector
   */
  virtual int getFeatureSetSize() = 0;

  virtual ~Classifier()
  {
  }

protected:
  /**
   * @brief printEval print evaluation results on the screen and save them in a file.
   * Measures: accuracy, macro precision, macro recall, macro f1-score
   * @param confusion_matrix Constructed based on the true and predicted label
   * @param eval_filename
   */
  void printEval(const cv::Mat& confusion_matrix, const std::string& eval_filename);

  ros::NodeHandle nh_;
  double validation_data_ratio_; /** ratio of training data used for validation */
};
}
#endif  // CLASSIFIERS_CLASSIFIER_H
