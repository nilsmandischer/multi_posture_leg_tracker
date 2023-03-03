#ifndef CLASSIFIERS_OPENCV_CLASSIFIER_H
#define CLASSIFIERS_OPENCV_CLASSIFIER_H

#include <classifier_lib/classifier.h>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

namespace multi_posture_leg_tracker {

/**
 * @brief The OpenCVClassifier class
 * OpenCV classifier interface
 */
class OpenCVClassifier : public Classifier
{
public:
  OpenCVClassifier(ros::NodeHandle& nh) : Classifier(nh)
  {
  }

  virtual void train(std::vector<std::vector<float>>& leg_feature_matrix,
                     std::vector<std::vector<float>>& squat_feature_matrix,
                     std::vector<std::vector<float>>& neg_feature_matrix, const std::string& eval_filename_path) = 0;
  virtual void classifyFeatureVector(const std::vector<float>& feature_vector, float& label, float& confidence) = 0;

  virtual int getFeatureSetSize() = 0;
  virtual bool loadModel(const std::string& filename) = 0;

  virtual bool saveModel(const std::string& filename);

protected:
  virtual cv::Ptr<cv::ml::StatModel> getStatModel() = 0;
  virtual void validate(const cv::Ptr<cv::ml::StatModel>& model, const cv::Mat& data, const cv::Mat& responses,
                        const std::string& filename);
};

}
#endif  // CLASSIFIERS_OPENCV_CLASSIFIER_H
