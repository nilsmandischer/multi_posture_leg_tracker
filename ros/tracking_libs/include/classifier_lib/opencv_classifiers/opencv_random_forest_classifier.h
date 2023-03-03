#ifndef CLASSIFIERS_OPENCV_RANDOM_FOREST_CLASSIFIER_H
#define CLASSIFIERS_OPENCV_RANDOM_FOREST_CLASSIFIER_H

#include <classifier_lib/opencv_classifiers/opencv_classifier.h>

namespace multi_posture_leg_tracker {

/**
 * @brief The OpenCVRandomForestClassifier class
 * Random forest algorithm in OpenCV library
 */
class OpenCVRandomForestClassifier : public OpenCVClassifier
{
public:
  OpenCVRandomForestClassifier(ros::NodeHandle& nh);

  virtual void train(std::vector<std::vector<float>>& leg_feature_matrix,
                     std::vector<std::vector<float>>& squat_feature_matrix,
                     std::vector<std::vector<float>>& neg_feature_matrix, const std::string& eval_filename_path);

  virtual bool loadModel(const std::string& filename);

  virtual cv::Ptr<cv::ml::StatModel> getStatModel();

  virtual int getFeatureSetSize();

  virtual void classifyFeatureVector(const std::vector<float>& feature_vector, float& label, float& confidence);

private:
  cv::Ptr<cv::ml::RTrees> forest_;
};
}
#endif  // CLASSIFIERS_OPENCV_RANDOM_FOREST_CLASSIFIER_H
