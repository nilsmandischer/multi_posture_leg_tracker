#include <classifier_lib/opencv_classifiers/opencv_random_forest_classifier.h>

namespace multi_posture_leg_tracker {

OpenCVRandomForestClassifier::OpenCVRandomForestClassifier(ros::NodeHandle& nh) : OpenCVClassifier(nh)
{
  forest_ = cv::ml::RTrees::create();
}

cv::Ptr<cv::ml::StatModel> OpenCVRandomForestClassifier::getStatModel()
{
  return forest_;
}

int OpenCVRandomForestClassifier::getFeatureSetSize()
{
  return forest_->getVarCount();
}

void OpenCVRandomForestClassifier::train(std::vector<std::vector<float>>& leg_feature_matrix,
                                         std::vector<std::vector<float>>& squat_feature_matrix,
                                         std::vector<std::vector<float>>& neg_feature_matrix,
                                         const std::string& eval_filename_path)
{
  int sample_size = leg_feature_matrix.size() + squat_feature_matrix.size() + neg_feature_matrix.size();
  int feature_dim = leg_feature_matrix[0].size();

  CvMat* cv_data = cvCreateMat(sample_size, feature_dim, CV_32FC1);  // 32-bit float and 1 channel
  CvMat* cv_resp = cvCreateMat(sample_size, 1, CV_32S);

  // Put leg data in opencv format.
  int j = 0;
  for (std::vector<std::vector<float>>::const_iterator i = leg_feature_matrix.begin(); i != leg_feature_matrix.end();
       i++)
  {
    float* data_row = (float*)(cv_data->data.ptr + cv_data->step * j);
    for (int k = 0; k < feature_dim; k++)
      data_row[k] = (*i)[k];

    cv_resp->data.i[j] = 1;
    j++;
  }

  // Put squat data in opencv format.
  for (std::vector<std::vector<float>>::const_iterator i = squat_feature_matrix.begin();
       i != squat_feature_matrix.end(); i++)
  {
    float* data_row = (float*)(cv_data->data.ptr + cv_data->step * j);
    for (int k = 0; k < feature_dim; k++)
      data_row[k] = (*i)[k];

    cv_resp->data.i[j] = 2;
    j++;
  }

  // Put negative data in opencv format.
  for (std::vector<std::vector<float>>::const_iterator i = neg_feature_matrix.begin(); i != neg_feature_matrix.end();
       i++)
  {
    float* data_row = (float*)(cv_data->data.ptr + cv_data->step * j);
    for (int k = 0; k < feature_dim; k++)
      data_row[k] = (*i)[k];

    cv_resp->data.i[j] = 0;
    j++;
  }

  CvMat* var_type = cvCreateMat(1, feature_dim + 1, CV_8U);
  cvSet(var_type, cvScalarAll(cv::ml::VAR_ORDERED));
  cvSetReal1D(var_type, feature_dim, cv::ml::VAR_CATEGORICAL);

  // Random forest training parameters
  float priors[] = { 1.0, static_cast<float>(neg_feature_matrix.size()) / static_cast<float>(leg_feature_matrix.size()),
                     static_cast<float>(neg_feature_matrix.size()) / static_cast<float>(squat_feature_matrix.size()) };
  cv::Mat priors_mat = cv::Mat(1, 3, CV_32F, priors);

  // SET PARAMETERS
  forest_->setMaxDepth(10);                  // max depth of tree (20)
  forest_->setMinSampleCount(5);             // min sample count to split tree (2)
  forest_->setRegressionAccuracy(0);         // regression accuracy (?)
  forest_->setUseSurrogates(false);          // use surrogates (?)
  forest_->setMaxCategories(1000);           // max categories
  forest_->setPriors(priors_mat);            // priors
  forest_->setCalculateVarImportance(true);  // calculate variable importance
  forest_->setActiveVarCount(15);            // number of active vars for each tree node (default from
                                             // scikit-learn is: (int)round(sqrt(feature_dim))
  int nTrees = 100;                          // max trees in forest (default of 10 from scikit-learn does worse)
  forest_->setRegressionAccuracy(0.001f);    // forest accuracy (sufficient OOB error)
  forest_->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, nTrees,
                                            1e-6));  // termination criteria. CV_TERMCRIT_ITER =
                                                     // once we reach max number of forests

  cv::Ptr<cv::ml::TrainData> data = cv::ml::TrainData::create(cv::cvarrToMat(cv_data),    // train data
                                                              cv::ml::ROW_SAMPLE,         // tflag
                                                              cv::cvarrToMat(cv_resp),    // responses (i.e. labels)
                                                              cv::noArray(),              // varldx (?)
                                                              cv::noArray(),              // sampleldx (?)
                                                              cv::noArray(),              // missing data mask
                                                              cv::cvarrToMat(var_type));  // variable type

  if (validation_data_ratio_ > 0)
  {
    data->setTrainTestSplitRatio(1 - validation_data_ratio_, true);  // true: shuffle data
    forest_->train(data);

    std::cout << "Number of trees: " << forest_->getRoots().size() << std::endl;

    cv::Mat train_samples = data->getTrainSamples();
    cv::Mat train_resp = data->getTrainResponses();
    cv::Mat test_samples = data->getTestSamples();
    cv::Mat test_resp = data->getTestResponses();

    ROS_INFO("Testing on training data:");

    std::string eval_file = eval_filename_path + "opencv_random_forest_" + std::to_string(feature_dim) + "_train.yaml";
    std::cout << eval_file << std::endl;
    validate(forest_, train_samples, train_resp, eval_file);

    ROS_INFO("Testing on validation data:");
    eval_file = eval_filename_path + "opencv_random_forest_" + std::to_string(feature_dim) + "_valid.yaml";
    validate(forest_, test_samples, test_resp, eval_file);

    float test_error = forest_->calcError(data, false, cv::noArray());  // true: Error on test data; y: predicted label
    std::cout << "Test error is: " << test_error << std::endl;
  }
  else
  {
    forest_->train(data);
    std::cout << "Number of trees: " << forest_->getRoots().size() << std::endl;

    cv::Mat train_samples = data->getTrainSamples();
    cv::Mat train_resp = data->getTrainResponses();
    ROS_INFO("Testing on training data:");
    std::string eval_file = eval_filename_path + "opencv_random_forest_" + std::to_string(feature_dim) + "_train.yaml";
    std::cout << eval_file << std::endl;
    validate(forest_, train_samples, train_resp, eval_file);
  }

  cvReleaseMat(&cv_data);
  cvReleaseMat(&cv_resp);
  cvReleaseMat(&var_type);

  // print feature importance
  cv::Mat var_importance = forest_->getVarImportance();
  if (!var_importance.empty())
  {
    double imp_sum = cv::sum(var_importance)[0];
    printf("variable importance (in %%):\n");
    for (int i = 0; i < var_importance.total(); i++)
      printf("%-2d\t%.4f\n", i + 1, 100.f * var_importance.at<float>(i) / imp_sum);
  }
}

bool OpenCVRandomForestClassifier::loadModel(const std::string& filename)
{
  forest_ = cv::ml::StatModel::load<cv::ml::RTrees>(filename);
}

void OpenCVRandomForestClassifier::classifyFeatureVector(const std::vector<float>& feature_vector, float& label,
                                                         float& confidence)
{
  int feature_dim = feature_vector.size();
  cv::Mat tmp_vec = cv::Mat(1, feature_dim, CV_32FC1);
  for (int i = 0; i < feature_dim; i++)
    tmp_vec.at<float>(0, i) = (float)(feature_vector[i]);

  label = forest_->predict(tmp_vec);

  float probability_of_leg;
  float probability_of_squat;

  cv::Mat result;
  forest_->getVotes(feature_vector, result, 0);
  int positive_votes = result.at<int>(1, 1);
  int negative_votes = result.at<int>(1, 0);
  int squat_votes = result.at<int>(1, 2);
  probability_of_leg = positive_votes / static_cast<double>(positive_votes + negative_votes + squat_votes);
  probability_of_squat = squat_votes / static_cast<double>(positive_votes + negative_votes + squat_votes);
  confidence = probability_of_leg + probability_of_squat;  // prob of human
}
}