#ifndef CLASSIFIERS_MLPACK_ADABOOST_CLASSIFIER_H
#define CLASSIFIERS_MLPACK_ADABOOST_CLASSIFIER_H

#include <classifier_lib/mlpack_classifiers/mlpack_classifier.h>

#include <mlpack/methods/adaboost/adaboost.hpp>
#include <mlpack/methods/adaboost/adaboost_model.hpp>

namespace multi_posture_leg_tracker {

/**
 * @brief The AdaBoostWrapper class
 * A wrapper of AdaBoost in mlpack to enable setting and loading feature set size
 */
class AdaBoostWrapper
{
public:
  AdaBoostWrapper() : dimensionality_(0)
  {
    boost_ = new mlpack::adaboost::AdaBoost<mlpack::tree::ID3DecisionStump>();
  }

  ~AdaBoostWrapper()
  {
    delete boost_;
  }

  void Train(const arma::mat& data, const arma::Row<size_t>& labels, const size_t numClasses,
             const arma::rowvec& weights, const size_t iterations, const double tolerance)
  {
    dimensionality_ = data.n_rows;
    mlpack::tree::ID3DecisionStump ds(data, labels, numClasses, weights);
    boost_->Train(data, labels, numClasses, ds, iterations, tolerance);
  }

  void Train(const arma::mat& data, const arma::Row<size_t>& labels, const size_t numClasses, const size_t iterations,
             const double tolerance)
  {
    dimensionality_ = data.n_rows;
    mlpack::tree::ID3DecisionStump ds(data, labels, numClasses);
    boost_->Train(data, labels, numClasses, ds, iterations, tolerance);
  }

  int getDimensionality() const
  {
    return dimensionality_;
  }

  // modify dimensionality
  int& setDimensionality()
  {
    return dimensionality_;
  }

  template <typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar& BOOST_SERIALIZATION_NVP(boost_);
    ar& BOOST_SERIALIZATION_NVP(dimensionality_);
  }

  mlpack::adaboost::AdaBoost<mlpack::tree::ID3DecisionStump>* boost_;

private:
  int dimensionality_;
};

/**
 * @brief The OneVsOneWrapper class to save and load three trained binary models
 * One-Vs-One for generalization to multiclass problem
 */
class OneVsOneWrapper
{
public:
  OneVsOneWrapper()
  {
  }

  template <typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar& BOOST_SERIALIZATION_NVP(abw1_);
    ar& BOOST_SERIALIZATION_NVP(abw2_);
    ar& BOOST_SERIALIZATION_NVP(abw3_);
  }

  AdaBoostWrapper* abw1_;
  AdaBoostWrapper* abw2_;
  AdaBoostWrapper* abw3_;
};

class MlpackAdaBoostClassifier : public MlpackClassifier
{
public:
  MlpackAdaBoostClassifier(ros::NodeHandle& nh);

  virtual void train(std::vector<std::vector<float>>& leg_feature_matrix,
                     std::vector<std::vector<float>>& squat_feature_matrix,
                     std::vector<std::vector<float>>& neg_feature_matrix, const std::string& eval_filename_path);
  virtual void classifyFeatureVector(const std::vector<float>& feature_vector, float& label, float& confidence);
  virtual int getFeatureSetSize();
  virtual bool loadModel(const std::string& filename);
  virtual bool saveModel(const std::string& filename);

  ~MlpackAdaBoostClassifier();

private:
  void classifiyArmaVector(const arma::mat& feature_vector, float& label, float& confidence);
  void validateBinaryClassifier(AdaBoostWrapper* model, const arma::mat& data, const arma::Row<size_t>& responses,
                                bool save = false, const std::string& filename = "");
  virtual void classifyArmaMat(const arma::mat& feature_mat, arma::Row<size_t>& predictions);

  AdaBoostWrapper* abw1_;
  AdaBoostWrapper* abw2_;
  AdaBoostWrapper* abw3_;
};
}
#endif  // CLASSIFIERS_MLPACK_ADABOOST_CLASSIFIER_H
