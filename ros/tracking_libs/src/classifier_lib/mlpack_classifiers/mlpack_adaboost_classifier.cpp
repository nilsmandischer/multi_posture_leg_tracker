#include <classifier_lib/mlpack_classifiers/mlpack_adaboost_classifier.h>

namespace multi_posture_leg_tracker {

MlpackAdaBoostClassifier::MlpackAdaBoostClassifier(ros::NodeHandle& nh) : MlpackClassifier(nh)
{
  abw1_ = new AdaBoostWrapper();
  abw2_ = new AdaBoostWrapper();
  abw3_ = new AdaBoostWrapper();
}

void MlpackAdaBoostClassifier::train(std::vector<std::vector<float>>& leg_feature_matrix,
                                     std::vector<std::vector<float>>& squat_feature_matrix,
                                     std::vector<std::vector<float>>& neg_feature_matrix,
                                     const std::string& eval_filename_path)
{
  size_t sample_size = leg_feature_matrix.size() + squat_feature_matrix.size() + neg_feature_matrix.size();
  size_t sample_size_1 = leg_feature_matrix.size() + squat_feature_matrix.size();
  size_t sample_size_2 = leg_feature_matrix.size() + neg_feature_matrix.size();
  size_t sample_size_3 = squat_feature_matrix.size() + neg_feature_matrix.size();
  int feature_dim = leg_feature_matrix[0].size();

  arma::mat data(feature_dim, sample_size);  // leg + squat + neg: for test
  arma::Row<size_t> resp(sample_size);
  arma::mat data1(feature_dim, sample_size_1);  // leg + squat
  arma::Row<size_t> resp1(sample_size_1);
  arma::mat data2(feature_dim, sample_size_2);  // leg + neg
  arma::Row<size_t> resp2(sample_size_2);
  arma::mat data3(feature_dim, sample_size_3);  // squat + neg
  arma::Row<size_t> resp3(sample_size_3);

  size_t m = 0;
  size_t j = 0;
  size_t k = 0;
  size_t l = 0;
  for (std::vector<std::vector<float>>::const_iterator i = leg_feature_matrix.begin(); i != leg_feature_matrix.end();
       i++)
  {
    data1.col(j) = arma::conv_to<arma::vec>::from(*i);
    resp1(j) = 0;
    j++;

    data2.col(l) = arma::conv_to<arma::vec>::from(*i);
    resp2(l) = 1;
    l++;

    data.col(m) = arma::conv_to<arma::vec>::from(*i);
    resp(m) = 1;
    m++;
  }

  for (std::vector<std::vector<float>>::const_iterator i = squat_feature_matrix.begin();
       i != squat_feature_matrix.end(); i++)
  {
    data1.col(j) = arma::conv_to<arma::vec>::from(*i);
    resp1(j) = 1;
    j++;

    data3.col(k) = arma::conv_to<arma::vec>::from(*i);
    resp3(k) = 1;
    k++;

    data.col(m) = arma::conv_to<arma::vec>::from(*i);
    resp(m) = 2;
    m++;
  }

  for (std::vector<std::vector<float>>::const_iterator i = neg_feature_matrix.begin(); i != neg_feature_matrix.end();
       i++)
  {
    data2.col(l) = arma::conv_to<arma::vec>::from(*i);
    resp2(l) = 0;
    l++;

    data3.col(k) = arma::conv_to<arma::vec>::from(*i);
    resp3(k) = 0;
    k++;

    data.col(m) = arma::conv_to<arma::vec>::from(*i);
    resp(m) = 0;
    m++;
  }

  const size_t num_classes = 2;
  const size_t iterations = 100;
  const double tolerance = 1e-6;

  if (validation_data_ratio_ > 0)
  {
    arma::mat train_data_1;
    arma::Row<size_t> train_label_1;
    arma::mat test_data_1;
    arma::Row<size_t> test_label_1;
    mlpack::math::RandomSeed(27);
    const double test_ratio = 0.2;
    mlpack::data::Split(data1, resp1, train_data_1, test_data_1, train_label_1, test_label_1, test_ratio, true);

    arma::mat train_data_2;
    arma::Row<size_t> train_label_2;
    arma::mat test_data_2;
    arma::Row<size_t> test_label_2;
    mlpack::math::RandomSeed(27);
    mlpack::data::Split(data2, resp2, train_data_2, test_data_2, train_label_2, test_label_2, test_ratio, true);

    arma::mat train_data_3;
    arma::Row<size_t> train_label_3;
    arma::mat test_data_3;
    arma::Row<size_t> test_label_3;
    mlpack::math::RandomSeed(27);
    mlpack::data::Split(data3, resp3, train_data_3, test_data_3, train_label_3, test_label_3, test_ratio, true);

    abw1_->Train(train_data_1, train_label_1, num_classes, iterations, tolerance);
    abw2_->Train(train_data_2, train_label_2, num_classes, iterations, tolerance);
    abw3_->Train(train_data_3, train_label_3, num_classes, iterations, tolerance);

    std::cout << "\n\nTesting Model1(leg vs squat) on validation data:" << std::endl;
    validateBinaryClassifier(abw1_, test_data_1, test_label_1);

    std::cout << "\n\nTesting Model2(bkgd vs leg) on validation data:" << std::endl;
    validateBinaryClassifier(abw2_, test_data_2, test_label_2);

    std::cout << "\n\nTesting Model3(bkgd vs squat) on validation data:" << std::endl;
    validateBinaryClassifier(abw3_, test_data_3, test_label_3);

    std::string eval_file = eval_filename_path + "mlpack_adaboost_" + std::to_string(feature_dim);
    ROS_INFO("Testing on training data:");
    std::string train_eval_file = eval_file + "_train.yaml";
    validate(data, resp, train_eval_file);
  }
  else
  {
    abw1_->Train(data1, resp1, num_classes, iterations, tolerance);
    abw2_->Train(data2, resp2, num_classes, iterations, tolerance);
    abw3_->Train(data3, resp3, num_classes, iterations, tolerance);

    std::cout << "\n\nTesting Model1(leg vs squat) on training data:" << std::endl;
    validateBinaryClassifier(abw1_, data1, resp1);

    std::cout << "\n\nTesting Model2(bkgd vs leg) on training data:" << std::endl;
    validateBinaryClassifier(abw2_, data2, resp2);

    std::cout << "\n\nTesting Model3(bkgd vs squat) on training data:" << std::endl;
    validateBinaryClassifier(abw3_, data3, resp3);

    std::string eval_file = eval_filename_path + "mlpack_adaboost_" + std::to_string(feature_dim);
    ROS_INFO("Testing on training data:");
    std::string train_eval_file = eval_file + "_train.yaml";
    validate(data, resp, train_eval_file);
  }
}

void MlpackAdaBoostClassifier::classifyFeatureVector(const std::vector<float>& feature_vector, float& label,
                                                     float& confidence)
{
  int feature_dim = feature_vector.size();
  arma::mat tmp_vec(feature_dim, 1, arma::fill::zeros);
  for (int i = 0; i < feature_dim; i++)
    tmp_vec(i) = static_cast<float>(feature_vector[i]);

  classifiyArmaVector(tmp_vec, label, confidence);
}

int MlpackAdaBoostClassifier::getFeatureSetSize()
{
  return abw1_->getDimensionality();
}

bool MlpackAdaBoostClassifier::loadModel(const std::string& filename)
{
  ROS_INFO_STREAM("Loading classifier model: " << filename);
  OneVsOneWrapper* ovow = new OneVsOneWrapper();
  bool load = mlpack::data::Load(filename.c_str(), "model", ovow);
  abw1_ = ovow->abw1_;
  abw2_ = ovow->abw2_;
  abw3_ = ovow->abw3_;
  return load;
}

bool MlpackAdaBoostClassifier::saveModel(const std::string& filename)
{
  std::string save_filename = filename + ".xml";
  ROS_INFO_STREAM("Saving classifier model: " << save_filename);
  OneVsOneWrapper* ovow = new OneVsOneWrapper();
  ovow->abw1_ = abw1_;
  ovow->abw2_ = abw2_;
  ovow->abw3_ = abw3_;
  bool save = mlpack::data::Save(save_filename.c_str(), "model", ovow, false);
  return save;
}

MlpackAdaBoostClassifier::~MlpackAdaBoostClassifier()
{
  delete abw1_;
  delete abw2_;
  delete abw3_;
}

void MlpackAdaBoostClassifier::classifiyArmaVector(const arma::mat& feature_vector, float& label, float& confidence)
{
  arma::Row<size_t> prediction1;
  arma::mat probability1;
  abw1_->boost_->Classify(feature_vector, prediction1, probability1);

  arma::Row<size_t> prediction2;
  arma::mat probability2;
  abw2_->boost_->Classify(feature_vector, prediction2, probability2);

  arma::Row<size_t> prediction3;
  arma::mat probability3;
  abw3_->boost_->Classify(feature_vector, prediction3, probability3);

  arma::vec prob(3, arma::fill::zeros);

  // sum the probability of winning class
  if (prediction1.at(0) == 0)
    prob.at(1) += probability1.at(0);
  else
    prob.at(2) += probability1.at(1);

  if (prediction2.at(0) == 0)
    prob.at(0) += probability2.at(0);
  else
    prob.at(1) += probability2.at(1);

  if (prediction3.at(0) == 0)
    prob.at(0) += probability3.at(0);
  else
    prob.at(2) += probability3.at(1);

  prob = arma::normalise(prob, 1);
  label = prob.index_max();
  confidence = prob.at(1) + prob.at(2);
}

void MlpackAdaBoostClassifier::validateBinaryClassifier(AdaBoostWrapper* model, const arma::mat& data,
                                                        const arma::Row<size_t>& responses, bool save,
                                                        const std::string& filename)
{
  arma::mat confusion_mat(2, 2, arma::fill::zeros);
  arma::Row<size_t> pre;
  model->boost_->Classify(data, pre);
  for (size_t i = 0; i < responses.n_cols; i++)
  {
    confusion_mat.at(responses.at(i), pre.at(i))++;
  }
  std::cout << "confusion matrix =" << std::endl << " " << confusion_mat << std::endl;

  double cvPrecision = mlpack::cv::Precision<mlpack::cv::Binary>::Evaluate(*(model->boost_), data, responses);
  std::cout << "Precision: " << cvPrecision;
  double cvRecall = mlpack::cv::Recall<mlpack::cv::Binary>::Evaluate(*(model->boost_), data, responses);
  std::cout << "\nRecall: " << cvRecall;
  double cvF1 = mlpack::cv::F1<mlpack::cv::Binary>::Evaluate(*(model->boost_), data, responses);
  std::cout << "\nF1: " << cvF1 << std::endl << std::endl;

  // Obtain weak classifiers (features) weights
  std::map<size_t, double> dim_weight_map;  // split dimension and corresponding weight

  for (size_t i = 0; i < model->boost_->WeakLearners(); i++)
  {
    size_t dim = model->boost_->WeakLearner(i).SplitDimension();
    auto dim_in_map = dim_weight_map.find(dim);
    if (dim_in_map != dim_weight_map.end())
      dim_in_map->second += model->boost_->Alpha(i);
    else
      dim_weight_map[dim] = model->boost_->Alpha(i);
  }

  for (const auto& pair : dim_weight_map)
    std::cout << "{" << pair.first + 1 << ", " << pair.second << "}" << std::endl;
}

void MlpackAdaBoostClassifier::classifyArmaMat(const arma::mat& feature_mat, arma::Row<size_t>& predictions)
{
  float label;
  float confidence;
  for (size_t i = 0; i < feature_mat.n_cols; i++)
  {
    classifiyArmaVector(feature_mat.col(i), label, confidence);
    predictions(i) = static_cast<size_t>(label);
  }
}
}