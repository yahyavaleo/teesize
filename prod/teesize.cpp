#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

class LandmarkDetector
{
private:
  torch::jit::script::Module model;
  torch::Device device;

  cv::Mat warp(const cv::Mat &image, const std::vector<cv::Point2f> &distorted_points,
              float true_width, float true_height, float pixeltoinch)
  {
    float scaled_width = true_width * pixeltoinch;
    float scaled_height = true_height * pixeltoinch;

    std::vector<cv::Point2f> fixed_points = {
        cv::Point2f(0, 0),
        cv::Point2f(scaled_width, 0),
        cv::Point2f(scaled_width, scaled_height),
        cv::Point2f(0, scaled_height)};

    cv::Mat M = cv::getPerspectiveTransform(distorted_points, fixed_points);
    cv::Mat warped;
    cv::warpPerspective(image, warped, M, cv::Size(scaled_width, scaled_height));
    return warped;
  }

  cv::Mat sharpen(const cv::Mat &image)
  {
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0,
                      -1, 5, -1,
                      0, -1, 0);
    cv::Mat sharpened;
    cv::filter2D(image, sharpened, -1, kernel);
    return sharpened;
  }

  cv::Mat trim(const cv::Mat &image, int margin)
  {
    return image(
        cv::Range(margin, image.rows - margin),
        cv::Range(margin, image.cols - margin));
  }

  std::vector<cv::Point2f> detectChessboard(const cv::Mat &original_image)
  {
    cv::Mat image_gray;
    cv::cvtColor(original_image, image_gray, cv::COLOR_BGR2GRAY);
    int height = image_gray.rows, width = image_gray.cols;

    std::vector<cv::Point2f> corners;
    std::vector<cv::Point2f> quadrant_corners;

    cv::Mat ul_image = image_gray(
        cv::Range(0, height / 2),
        cv::Range(0, width / 2));
    cv::findChessboardCorners(ul_image, cv::Size(5, 5), corners);
    quadrant_corners.push_back(corners[12]);

    cv::Mat ur_image = image_gray(
        cv::Range(0, height / 2),
        cv::Range(width / 2, width));
    cv::findChessboardCorners(ur_image, cv::Size(5, 5), corners);
    quadrant_corners.push_back(corners[12] + cv::Point2f(width / 2, 0));

    cv::Mat br_image = image_gray(
        cv::Range(height / 2, height),
        cv::Range(width / 2, width));
    cv::findChessboardCorners(br_image, cv::Size(5, 5), corners);
    quadrant_corners.push_back(corners[12] + cv::Point2f(width / 2, height / 2));

    cv::Mat bl_image = image_gray(
        cv::Range(height / 2, height),
        cv::Range(0, width / 2));
    cv::findChessboardCorners(bl_image, cv::Size(5, 5), corners);
    quadrant_corners.push_back(corners[12] + cv::Point2f(0, height / 2));

    return quadrant_corners;
  }

  cv::Mat perspectiveCorrection(const cv::Mat &image, float true_width, float true_height,
                                float pixeltoinch, int margin, bool is_blur = false)
  {
    auto points = detectChessboard(image);
    cv::Mat warped = warp(image, points, true_width, true_height, pixeltoinch);

    if (is_blur)
    {
      warped = sharpen(warped);
    }

    return trim(warped, margin);
  }

  torch::Tensor softmax2D(torch::Tensor x)
  {
    auto exp_y = torch::exp(x);
    return exp_y / torch::sum(exp_y, {2, 3}, true);
  }

  cv::Point2f getHottestPoint(const cv::Mat &heatmap)
  {
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(heatmap, &minVal, &maxVal, &minLoc, &maxLoc);
    return cv::Point2f(maxLoc.y, maxLoc.x);
  }

  std::vector<cv::Point2f> getLandmarks(const torch::Tensor &heatmaps)
  {
    auto heatmaps_acc = heatmaps.accessor<float, 4>();
    std::vector<cv::Point2f> landmarks;

    for (int i = 0; i < heatmaps.size(1); ++i)
    {
      cv::Mat single_heatmap(heatmaps.size(2), heatmaps.size(3), CV_32F);

      for (int y = 0; y < heatmaps.size(2); ++y)
      {
        for (int x = 0; x < heatmaps.size(3); ++x)
        {
          single_heatmap.at<float>(y, x) = heatmaps_acc[0][i][y][x];
        }
      }

      cv::Point2f hottest_point = getHottestPoint(single_heatmap);
      landmarks.push_back(cv::Point2f(hottest_point.y, hottest_point.x));
    }

    return landmarks;
  }

public:
  LandmarkDetector(const std::string &model_path) : device(torch::kCPU)
  {
    try
    {
      model = torch::jit::load(model_path);
      model.to(device);
      model.eval();
    }
    catch (const c10::Error &e)
    {
      std::cerr << "Error loading the model: " << e.what() << std::endl;
      exit(1);
    }
  }

  std::vector<cv::Point2f> predict(const std::string &image_path)
  {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
      throw std::runtime_error("Could not read image");
    }

    const int W = 256, H = 256;
    float aspect_ratio = static_cast<float>(W) / H;

    cv::Mat padded, resized;
    cv::resize(image, resized, cv::Size(W, H));

    torch::Tensor tensor_image = torch::from_blob(resized.data, {1, 1, resized.rows, resized.cols}, torch::kByte);
    tensor_image = tensor_image.to(torch::kFloat32).div(255.0);
    tensor_image = tensor_image.to(device);

    torch::NoGradGuard no_grad;
    auto output = model.forward({tensor_image}).toTensor();
    output = softmax2D(output);

    output = output.to(torch::kCPU);

    return getLandmarks(output);
  }

  cv::Mat processImage(const std::string &input_path, const std::string &output_path)
  {
    cv::Mat image = cv::imread(input_path);

    cv::Mat corrected_image = perspectiveCorrection(image, 60, 60, 10, 25);

    int height = corrected_image.rows, width = corrected_image.cols;
    int original_size = std::max(width, height);
    float scale_factor = static_cast<float>(original_size) / 256;

    cv::imwrite(output_path, corrected_image);
    return corrected_image;
  }
};

int main()
{
  try
  {
    const std::string input_path = "assets/shirt.png";
    const std::string output_path = "assets/corrected.png";
    const std::string model_path = "resnet.pt";

    LandmarkDetector detector(model_path);

    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat corrected_image = detector.processImage(input_path, output_path);
    std::vector<cv::Point2f> landmarks = detector.predict(output_path);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
    std::cout << "Landmarks:" << std::endl;

    for (const auto &landmark : landmarks)
    {
      std::cout << "(" << landmark.x << ", " << landmark.y << ")" << std::endl;
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}