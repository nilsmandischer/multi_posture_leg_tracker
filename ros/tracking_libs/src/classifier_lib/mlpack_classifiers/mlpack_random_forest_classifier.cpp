#include <classifier_lib/mlpack_classifiers/mlpack_random_forest_classifier.h>

namespace multi_posture_leg_tracker {

MlpackRandomForestClassifier::MlpackRandomForestClassifier(ros::NodeHandle& nh) : MlpackClassifier(nh)
{
  rfw_ = new RandomForestWrapper();
}

void MlpackRandomForestClassifier::train(std::vector<std::vector<float>>& leg_feature_matrix,
                                         std::vector<std::vector<float>>& squat_feature_matrix,
                                         std::vector<std::vector<float>>& neg_feature_matrix,
                                         const std::string& eval_filename_path)
{
  size_t sample_size = leg_feature_matrix.size() + squat_feature_matrix.size() + neg_feature_matrix.size();
  int feature_dim = leg_feature_matrix[0].size();
  rfw_->setDimensionality() = feature_dim;

  arma::mat data(feature_dim, sample_size);
  arma::Row<size_t> resp(sample_size);

  arma::rowvec weights_all(sample_size, arma::fill::ones);
  double leg_weight = neg_feature_matrix.size() / leg_feature_matrix.size();
  double squat_weight = neg_feature_matrix.size() / squat_feature_matrix.size();
  int j = 0;
  for (std::vector<std::vector<float>>::const_iterator i = leg_feature_matrix.begin(); i != leg_feature_matrix.end();
       i++)
  {
    data.col(j) = arma::conv_to<arma::vec>::from(*i);
    resp(j) = 1;
    weights_all(j) = leg_weight;
    j++;
  }

  for (std::vector<std::vector<float>>::const_iterator i = squat_feature_matrix.begin();
       i != squat_feature_matrix.end(); i++)
  {
    data.col(j) = arma::conv_to<arma::vec>::from(*i);
    resp(j) = 2;
    weights_all(j) = squat_weight;
    j++;
  }

  for (std::vector<std::vector<float>>::const_iterator i = neg_feature_matrix.begin(); i != neg_feature_matrix.end();
       i++)
  {
    data.col(j) = arma::conv_to<arma::vec>::from(*i);
    resp(j) = 0;
    j++;
  }

  const size_t num_classes = 3;
  const size_t num_trees = 100;
  const size_t min_leaf_size = 5;
  const double min_gain_split = 1e-7;
  const size_t max_depth = 10;
  const size_t random_dims = 15;
  mlpack::tree::MultipleRandomDimensionSelect mrds(random_dims);

  if (validation_data_ratio_ > 0)
  {
    arma::mat train_data;
    arma::Row<size_t> train_label;
    arma::mat test_data;
    arma::Row<size_t> test_label;
    mlpack::math::RandomSeed(27);
    const double test_ratio = validation_data_ratio_;
    mlpack::data::Split(data, resp, train_data, test_data, train_label, test_label, test_ratio, true);

    arma::rowvec weights(train_data.n_cols, arma::fill::ones);
    for (size_t i = 0; i < weights.n_cols; i++)
    {
      size_t label = train_label.at(i);
      if (label == 1)
        weights.at(i) = leg_weight;
      else if (label == 2)
        weights.at(i) = squat_weight;
    }

    rfw_->rf_.Train(train_data, train_label, num_classes, weights, num_trees, min_leaf_size, min_gain_split, max_depth,
                    mrds);

    ROS_INFO("Testing on training data:");
    std::string eval_file = eval_filename_path + "mlpack_random_forest_" + std::to_string(feature_dim) + "_train.yaml";
    std::cout << eval_file << std::endl;
    validate(train_data, train_label, eval_file);

    ROS_INFO("Testing on validation data:");
    eval_file = eval_filename_path + "mlpack_random_forest_" + std::to_string(feature_dim) + "_valid.yaml";
    validate(test_data, test_label, eval_file);

    double macro_precision = mlpack::cv::Precision<mlpack::cv::Macro>::Evaluate(rfw_->rf_, test_data, test_label);
    std::cout << "\nMacro Precision: " << macro_precision;
    double macro_recall = mlpack::cv::Recall<mlpack::cv::Macro>::Evaluate(rfw_->rf_, test_data, test_label);
    std::cout << "\nMacro Recall: " << macro_recall;
    double macro_f1 = mlpack::cv::F1<mlpack::cv::Macro>::Evaluate(rfw_->rf_, test_data, test_label);
    std::cout << "\nMacro F1: " << macro_f1;
    double accuracy = mlpack::cv::Accuracy::Evaluate(rfw_->rf_, test_data, test_label);
    std::cout << "\nAccuracy: " << accuracy;
    double micro_f1 = mlpack::cv::F1<mlpack::cv::Micro>::Evaluate(rfw_->rf_, test_data, test_label);
    std::cout << "\nMicro F1: " << micro_f1 << std::endl << std::endl;
  }
  else
  {
    arma::rowvec weights(data.n_cols, arma::fill::ones);
    for (size_t i = 0; i < weights.n_cols; i++)
    {
      size_t label = resp.at(i);
      if (label == 1)
        weights.at(i) = leg_weight;
      else if (label == 2)
        weights.at(i) = squat_weight;
    }

    rfw_->rf_.Train(data, resp, num_classes, num_trees, min_leaf_size, min_gain_split, max_depth, mrds);

    ROS_INFO("Testing on training data:");

    std::string eval_file = eval_filename_path + "mlpack_random_forest_" + std::to_string(feature_dim) + "_train.yaml";
    std::cout << eval_file << std::endl;
    validate(data, resp, eval_file);

    double macro_precision = mlpack::cv::Precision<mlpack::cv::Macro>::Evaluate(rfw_->rf_, data, resp);
    std::cout << "\nMacro Precision: " << macro_precision;
    double macro_recall = mlpack::cv::Recall<mlpack::cv::Macro>::Evaluate(rfw_->rf_, data, resp);
    std::cout << "\nMacro Recall: " << macro_recall;
    double macro_f1 = mlpack::cv::F1<mlpack::cv::Macro>::Evaluate(rfw_->rf_, data, resp);
    std::cout << "\nMacro F1: " << macro_f1;
    double accuracy = mlpack::cv::Accuracy::Evaluate(rfw_->rf_, data, resp);
    std::cout << "\nAccuracy: " << accuracy;
    double micro_f1 = mlpack::cv::F1<mlpack::cv::Micro>::Evaluate(rfw_->rf_, data, resp);
    std::cout << "\nMicro F1: " << micro_f1 << std::endl << std::endl;
  }
}

void MlpackRandomForestClassifier::classifyFeatureVector(const std::vector<float>& feature_vector, float& label,
                                                         float& confidence)
{
  int feature_dim = feature_vector.size();
  arma::vec tmp_vec(feature_dim, arma::fill::zeros);
  for (int i = 0; i < feature_dim; i++)
    tmp_vec(i) = static_cast<float>(feature_vector[i]);

  size_t prediction;
  arma::vec probability;
  rfw_->rf_.Classify(tmp_vec, prediction, probability);

  label = prediction;
  confidence = probability(1) + probability(2);
}

int MlpackRandomForestClassifier::getFeatureSetSize()
{
  return rfw_->getDimensionality();
}

bool MlpackRandomForestClassifier::loadModel(const std::string& filename)
{
  ROS_INFO_STREAM("Loading classifier model: " << filename);
  rfw_ = new RandomForestWrapper();
  bool load = mlpack::data::Load(filename.c_str(), "model", rfw_);
  return load;
}

bool MlpackRandomForestClassifier::saveModel(const std::string& filename)
{
  std::string save_filename = filename + ".xml";
  ROS_INFO_STREAM("Saving classifier model: " << save_filename);
  bool save = mlpack::data::Save(save_filename.c_str(), "model", rfw_, false);
  return save;
}

void MlpackRandomForestClassifier::classifyArmaMat(const arma::mat& feature_mat, arma::Row<size_t>& predictions)
{
  rfw_->rf_.Classify(feature_mat, predictions);
}
}