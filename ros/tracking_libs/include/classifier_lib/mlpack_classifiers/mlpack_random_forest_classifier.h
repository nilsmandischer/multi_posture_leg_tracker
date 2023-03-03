#ifndef CLASSIFIERS_MLPACK_RANDOM_FOREST_CLASSIFIER_H
#define CLASSIFIERS_MLPACK_RANDOM_FOREST_CLASSIFIER_H

#include <classifier_lib/mlpack_classifiers/mlpack_classifier.h>

#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>

namespace multi_posture_leg_tracker {

/**
 * @brief The RandomForestWrapper class
 * A wrapper of random forest in mlpack to enable saving and loading feature set size
 */
class RandomForestWrapper
{
public:
  RandomForestWrapper()
  {
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
    ar& BOOST_SERIALIZATION_NVP(rf_);
    ar& BOOST_SERIALIZATION_NVP(dimensionality_);
  }

  mlpack::tree::RandomForest<mlpack::tree::GiniGain, mlpack::tree::MultipleRandomDimensionSelect> rf_;

private:
  int dimensionality_;
};

/**
 * @brief The MlpackRandomForestClassifier class
 * Random forest algorithm in mlpack library
 */
class MlpackRandomForestClassifier : public MlpackClassifier
{
public:
  MlpackRandomForestClassifier(ros::NodeHandle& nh);

  virtual void train(std::vector<std::vector<float>>& leg_feature_matrix,
                     std::vector<std::vector<float>>& squat_feature_matrix,
                     std::vector<std::vector<float>>& neg_feature_matrix, const std::string& eval_filename_path);
  virtual void classifyFeatureVector(const std::vector<float>& feature_vector, float& label, float& confidence);
  virtual int getFeatureSetSize();
  virtual bool loadModel(const std::string& filename);
  virtual bool saveModel(const std::string& filename);

private:
  virtual void classifyArmaMat(const arma::mat& feature_mat, arma::Row<size_t>& predictions);

  RandomForestWrapper* rfw_;
};
}
#endif  // CLASSIFIERS_MLPACK_RANDOM_FOREST_CLASSIFIER_H
