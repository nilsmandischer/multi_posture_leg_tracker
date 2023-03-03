#include <classifier_lib/opencv_classifiers/opencv_classifier.h>

namespace multi_posture_leg_tracker {
  
bool OpenCVClassifier::saveModel(const std::string& filename)
{
  std::string save_filename = filename + ".yaml";
  ROS_INFO_STREAM("Saving classifier model: " << save_filename);
  getStatModel()->save(save_filename.c_str());
  return true;
}

void OpenCVClassifier::validate(const cv::Ptr<cv::ml::StatModel>& model, const cv::Mat& data, const cv::Mat& responses,
                                const std::string& filename)
{
  int num_samples = data.rows;
  cv::Mat confusion_mat = cv::Mat::zeros(3, 3, CV_32FC1);
  for (int i = 0; i < num_samples; i++)
  {
    cv::Mat sample = data.row(i);
    float pre = model->predict(sample);
    int resp = responses.at<int>(i);
    confusion_mat.at<float>(resp, (int)pre)++;
  }
  printEval(confusion_mat, filename);
}
}