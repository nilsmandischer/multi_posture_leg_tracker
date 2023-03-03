#ifndef CLASSIFIERS_MLPACK_CLASSIFIER_H
#define CLASSIFIERS_MLPACK_CLASSIFIER_H

#include <classifier_lib/classifier.h>

#include <mlpack/core.hpp>

#include <mlpack/core/cv/simple_cv.hpp>
#include <mlpack/core/cv/k_fold_cv.hpp>
#include <mlpack/core/cv/metrics/accuracy.hpp>
#include <mlpack/core/cv/metrics/precision.hpp>
#include <mlpack/core/cv/metrics/recall.hpp>
#include <mlpack/core/cv/metrics/f1.hpp>

#include <mlpack/core/hpt/cv_function.hpp>
#include <mlpack/core/hpt/hpt.hpp>

#include <mlpack/core/data/confusion_matrix.hpp>
#include <mlpack/core/data/split_data.hpp>

#include <opencv2/core/core.hpp>

namespace multi_posture_leg_tracker {

/**
 * @brief The MlpackClassifier class
 * Mlpack classifier interface
 */
class MlpackClassifier : public Classifier
{
public:
  MlpackClassifier(ros::NodeHandle& nh) : Classifier(nh)
  {
  }

  virtual void train(std::vector<std::vector<float>>& leg_feature_matrix,
                     std::vector<std::vector<float>>& squat_feature_matrix,
                     std::vector<std::vector<float>>& neg_feature_matrix, const std::string& eval_filename_path) = 0;
  virtual void classifyFeatureVector(const std::vector<float>& feature_vector, float& label, float& confidence) = 0;

  virtual bool loadModel(const std::string& filename) = 0;

  virtual int getFeatureSetSize() = 0;

  virtual bool saveModel(const std::string& filename) = 0;

protected:
  virtual void classifyArmaMat(const arma::mat& feature_mat, arma::Row<size_t>& predictions) = 0;

  virtual void validate(const arma::mat& data, const arma::Row<size_t>& responses, const std::string& filename)
  {
    const size_t num_classes = 3;
    arma::Row<size_t> predictions(data.n_cols);

    classifyArmaMat(data, predictions);

    arma::mat confusion_matrix;
    mlpack::data::ConfusionMatrix(predictions, responses, confusion_matrix, num_classes);
    cv::Mat opencv_mat(num_classes, num_classes, CV_32FC1);

    for (size_t i = 0; i < num_classes; i++)
    {
      for (size_t j = 0; j < num_classes; j++)
      {
        opencv_mat.at<float>(i, j) = confusion_matrix.at(j, i);
      }
    }

    printEval(opencv_mat, filename);
  }
};
}
#endif  // CLASSIFIERS_MLPACK_CLASSIFIER_H
