#include <classifier_lib/opencv_classifiers/opencv_adaboost_classifier.h>

namespace multi_posture_leg_tracker {

OpenCVAdaBoostClassifier::OpenCVAdaBoostClassifier(ros::NodeHandle& nh) : OpenCVClassifier(nh)
{
  boost_ = cv::ml::Boost::create();
}

void OpenCVAdaBoostClassifier::train(std::vector<std::vector<float>>& leg_feature_matrix,
                                     std::vector<std::vector<float>>& squat_feature_matrix,
                                     std::vector<std::vector<float>>& neg_feature_matrix,
                                     const std::string& eval_filename_path)
{
  int sample_size = leg_feature_matrix.size() + neg_feature_matrix.size() + squat_feature_matrix.size();
  int feature_dim = leg_feature_matrix[0].size();

  CvMat* cv_data = cvCreateMat(sample_size * 3, feature_dim + 1, CV_32FC1);  // 32-bit float and 1 channel
  CvMat* cv_resp = cvCreateMat(sample_size * 3, 1, CV_32S);

  // Put leg data in opencv format.
  int j = 0;
  for (std::vector<std::vector<float>>::const_iterator i = leg_feature_matrix.begin(); i != leg_feature_matrix.end();
       i++)
  {
    for (int c = 0; c < 3; c++)
    {
      float* data_row = (float*)(cv_data->data.ptr + cv_data->step * j);
      for (int k = 0; k < feature_dim; k++)
        data_row[k] = (*i)[k];
      data_row[feature_dim] = (float)c;

      if (c == 1)
        cv_resp->data.i[j] = 1;
      else
        cv_resp->data.i[j] = 0;

      j++;
    }
  }

  // Put squat data in opencv format.
  for (std::vector<std::vector<float>>::const_iterator i = squat_feature_matrix.begin();
       i != squat_feature_matrix.end(); i++)
  {
    for (int c = 0; c < 3; c++)
    {
      float* data_row = (float*)(cv_data->data.ptr + cv_data->step * j);
      for (int k = 0; k < feature_dim; k++)
        data_row[k] = (*i)[k];
      data_row[feature_dim] = (float)c;

      if (c == 2)
        cv_resp->data.i[j] = 1;
      else
        cv_resp->data.i[j] = 0;

      j++;
    }
  }

  // Put negative data in opencv format.
  for (std::vector<std::vector<float>>::const_iterator i = neg_feature_matrix.begin(); i != neg_feature_matrix.end();
       i++)
  {
    for (int c = 0; c < 3; c++)
    {
      float* data_row = (float*)(cv_data->data.ptr + cv_data->step * j);
      for (int k = 0; k < feature_dim; k++)
        data_row[k] = (*i)[k];
      data_row[feature_dim] = (float)c;

      if (c == 0)
        cv_resp->data.i[j] = 1;
      else
        cv_resp->data.i[j] = 0;

      j++;
    }
  }

  CvMat* var_type = cvCreateMat(1, feature_dim + 2, CV_8U);
  cvSet(var_type, cvScalarAll(cv::ml::VAR_ORDERED));
  cvSetReal1D(var_type, feature_dim, cv::ml::VAR_CATEGORICAL);
  cvSetReal1D(var_type, feature_dim + 1, cv::ml::VAR_CATEGORICAL);

  float priors[] = { 1.0, 2.0 };
  cv::Mat priors_mat = cv::Mat(1, 2, CV_32F, priors);

  boost_->setBoostType(cv::ml::Boost::REAL);
  boost_->setWeakCount(20);
  boost_->setWeightTrimRate(1.0);
  boost_->setUseSurrogates(false);
  boost_->setPriors(priors_mat);

  boost_->setMaxDepth(5);
  boost_->setMinSampleCount(5);
  boost_->setRegressionAccuracy(0.001f);

  cv::Ptr<cv::ml::TrainData> data = cv::ml::TrainData::create(cv::cvarrToMat(cv_data),    // train data
                                                              cv::ml::ROW_SAMPLE,         // tflag
                                                              cv::cvarrToMat(cv_resp),    // responses (i.e. labels)
                                                              cv::noArray(),              // varldx (?)
                                                              cv::noArray(),              // sampleldx (?)
                                                              cv::noArray(),              // missing data mask
                                                              cv::cvarrToMat(var_type));  // variable type

  boost_->train(data);

  std::string eval_file = eval_filename_path + "opencv_adaboost_" + std::to_string(feature_dim);
  ROS_INFO("Testing on training data:");
  std::string train_eval_file = eval_file + "_train.yaml";
  test(leg_feature_matrix, squat_feature_matrix, neg_feature_matrix, train_eval_file);

  cvReleaseMat(&cv_data);
  cvReleaseMat(&cv_resp);
  cvReleaseMat(&var_type);
}

bool OpenCVAdaBoostClassifier::loadModel(const std::string& filename)
{
  boost_ = cv::ml::StatModel::load<cv::ml::Boost>(filename);
}

cv::Ptr<cv::ml::StatModel> OpenCVAdaBoostClassifier::getStatModel()
{
  return boost_;
}

int OpenCVAdaBoostClassifier::getFeatureSetSize()
{
  // The last feature is an additional class indicator attribute which should be ignored.
  return boost_->getVarCount() - 1;
}

void OpenCVAdaBoostClassifier::classifyFeatureVector(const std::vector<float>& feature_vector, float& label,
                                                     float& confidence)
{
  int feature_dim = feature_vector.size();
  cv::Mat tmp_vec = cv::Mat(1, feature_dim + 1, CV_32FC1);
  for (int i = 0; i < feature_dim; i++)
    tmp_vec.at<float>(0, i) = (float)(feature_vector[i]);

  float pro[3];
  double sum = 0.0;
  float max = -FLT_MAX;
  for (int c = 0; c < 3; c++)
  {
    tmp_vec.at<float>(0, feature_dim) = (float)c;
    float pre = boost_->predict(tmp_vec, cv::noArray(), cv::ml::Boost::PREDICT_SUM);
    // Convert prediction to probabilty ( similar to logistic sigmoid function ) [Fotiadis et al.]
    pro[c] = 1 / (1 + exp(-2.0 * pre - 17.0));
    sum += pro[c];
    if (pre > max)
    {
      max = pre;
      label = c;
    }
  }
  float probability_of_leg = pro[1] / sum;
  float probability_of_squat = pro[2] / sum;
  if (!std::isnan(probability_of_leg) && !std::isnan(probability_of_squat))
    confidence = probability_of_leg + probability_of_squat;
  else
    confidence = std::abs(label - 0) > FLT_EPSILON;
}

}