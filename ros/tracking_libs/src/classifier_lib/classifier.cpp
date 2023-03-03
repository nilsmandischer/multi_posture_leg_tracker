#include <classifier_lib/classifier.h>

namespace multi_posture_leg_tracker {

Classifier::Classifier(ros::NodeHandle& nh) : nh_(nh)
{
  nh_.param("validation_data_ratio", validation_data_ratio_, 0.2);
}

void Classifier::test(std::vector<std::vector<float>>& leg_feature_matrix,
                      std::vector<std::vector<float>>& squat_feature_matrix,
                      std::vector<std::vector<float>>& neg_feature_matrix, const std::string& eval_filename)
{
  float pre;
  float confidence;
  cv::Mat confusion_mat = cv::Mat::zeros(3, 3, CV_32FC1);

  // construct confusion matrix
  for (std::vector<std::vector<float>>::const_iterator i = neg_feature_matrix.begin(); i != neg_feature_matrix.end();
       i++)
  {
    classifyFeatureVector(*i, pre, confidence);
    confusion_mat.at<float>(0, (int)pre)++;
  }
  for (std::vector<std::vector<float>>::const_iterator i = leg_feature_matrix.begin(); i != leg_feature_matrix.end();
       i++)
  {
    classifyFeatureVector(*i, pre, confidence);
    confusion_mat.at<float>(1, (int)pre)++;
  }
  for (std::vector<std::vector<float>>::const_iterator i = squat_feature_matrix.begin();
       i != squat_feature_matrix.end(); i++)
  {
    classifyFeatureVector(*i, pre, confidence);
    confusion_mat.at<float>(2, (int)pre)++;
  }

  printEval(confusion_mat, eval_filename);
}

void Classifier::printEval(const cv::Mat& confusion_mat, const std::string& eval_filename)
{
  int num_true_neg = 0;
  int num_true_leg = 0;
  int num_true_squat = 0;
  int num_pre_neg, num_pre_leg, num_pre_squat;
  num_pre_neg = num_pre_leg = num_pre_squat = 0;
  for (int c = 0; c < 3; c++)
  {
    num_true_neg += confusion_mat.at<float>(0, c);
    num_pre_neg += confusion_mat.at<float>(c, 0);

    num_true_leg += confusion_mat.at<float>(1, c);
    num_pre_leg += confusion_mat.at<float>(c, 1);

    num_true_squat += confusion_mat.at<float>(2, c);
    num_pre_squat += confusion_mat.at<float>(c, 2);
  }

  float neg_precision = confusion_mat.at<float>(0, 0) / num_pre_neg;
  float neg_recall = confusion_mat.at<float>(0, 0) / num_true_neg;
  float neg_f1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall);
  printf("Class Negative: \n");
  std::cout << "Precision: " << neg_precision << std::endl
            << "Recall: " << neg_recall << std::endl
            << "F1 Score: " << neg_f1 << std::endl
            << std::endl;

  float leg_precision = confusion_mat.at<float>(1, 1) / num_pre_leg;
  float leg_recall = confusion_mat.at<float>(1, 1) / num_true_leg;
  float leg_f1 = 2 * (leg_precision * leg_recall) / (leg_precision + leg_recall);
  printf("Class Leg: \n");
  std::cout << "Precision: " << leg_precision << std::endl
            << "Recall: " << leg_recall << std::endl
            << "F1 Score: " << leg_f1 << std::endl
            << std::endl;

  float squat_precision = confusion_mat.at<float>(2, 2) / num_pre_squat;
  float squat_recall = confusion_mat.at<float>(2, 2) / num_true_squat;
  float squat_f1 = 2 * (squat_precision * squat_recall) / (squat_precision + squat_recall);
  printf("Class Squat: \n");
  std::cout << "Precision: " << squat_precision << std::endl
            << "Recall: " << squat_recall << std::endl
            << "F1 Score: " << squat_f1 << std::endl
            << std::endl;

  std::cout << "confusion matrix =" << std::endl << " " << confusion_mat << std::endl << std::endl;

  float macro_precision = (neg_precision + leg_precision + squat_precision) / 3;
  float macro_recall = (neg_recall + leg_recall + squat_recall) / 3;
  // two ways to calculate macro f1 score:
  //  float macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall);
  float macro_f1 = (neg_f1 + leg_f1 + squat_f1) / 3;  // more common used

  float accuracy = (confusion_mat.at<float>(0, 0) + confusion_mat.at<float>(1, 1) + confusion_mat.at<float>(2, 2)) /
                   (num_true_neg + num_true_leg + num_true_squat);  // = micro_recall = micro_precision = micro_f1

  // storage the evaluation values
  cv::FileStorage fs(eval_filename, cv::FileStorage::WRITE);
  fs << "Class Negative";
  fs << "{"
     << "Precision" << neg_precision;
  fs << "Recall" << neg_recall;
  fs << "F1 Score" << neg_f1 << "}";

  fs << "Class Leg";
  fs << "{"
     << "Precision" << leg_precision;
  fs << "Recall" << leg_recall;
  fs << "F1 Score" << leg_f1 << "}";

  fs << "Class Squat";
  fs << "{"
     << "Precision" << squat_precision;
  fs << "Recall" << squat_recall;
  fs << "F1 Score" << squat_f1 << "}";

  fs << "confusion matrix" << confusion_mat;

  confusion_mat.row(0) /= num_true_neg;
  confusion_mat.row(1) /= num_true_leg;
  confusion_mat.row(2) /= num_true_squat;
  std::cout << "confusion matrix =" << std::endl << " " << confusion_mat << std::endl << std::endl;

  fs << "confusion matrix" << confusion_mat;

  fs << "Macro Precision" << macro_precision;
  fs << "Macro Recall" << macro_recall;
  fs << "Macro F1 Score" << macro_f1;
  fs << "Accuracy" << accuracy;
  fs.release();

  std::cout << "Macro Precision: " << macro_precision << std::endl
            << "Macro Recall: " << macro_recall << std::endl
            << "Macro F1 Score: " << macro_f1 << std::endl;
  std::cout << "Accuracy: " << accuracy << std::endl;

  std::cout << "The evaluation file has been saved to: " << eval_filename.c_str() << std::endl << std::endl;
}
}