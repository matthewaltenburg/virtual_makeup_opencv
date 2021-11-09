#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>

using namespace cv;
using namespace std;
using namespace dlib;

int main()
{
  // Main window name
  String windowName = "Main Window"; // Name of the main working window
  namedWindow(windowName, WINDOW_NORMAL);

  // file paths
  String girl = "girl.jpg";
  String imputImg = girl;

  // images
  Mat image = imread(imputImg); // Main image
  array2d<unsigned char> image_gray;
  load_image(image_gray, imputImg); // dlib image
  // mask
  Mat mask = Mat::zeros(image.size(), image.type());
  // Glasses image
  Mat image_glasses = imread("glasses.png", IMREAD_UNCHANGED);
  Mat glasses_split[4];
  cv::split(image_glasses, glasses_split);
  std::vector<Mat> channels;
  channels.push_back(glasses_split[3]);
  channels.push_back(glasses_split[3]);
  channels.push_back(glasses_split[3]);
  cv::merge(channels, image_glasses);

  // HSV image
  Mat imgHSV;
  cv::cvtColor(image, imgHSV, COLOR_BGR2HSV);

  // // testing dlib image
  // image_window win;
  // win.clear_overlay();
  // win.set_image(image_gray);
  // Checks for failure

  if (image.empty())
  {
    cout << "Could not open or find the image please add corret file path to image." << endl;
    return -1;
  }

  // create face detector abject
  frontal_face_detector faceDetector = get_frontal_face_detector();

  // find faces in image
  std::vector<dlib::rectangle> faces = faceDetector(image_gray);
  // // Testing image
  // image_window win;
  // win.clear_overlay();
  // win.set_image(image_gray);
  // win.add_overlay(faces, rgb_pixel(255,0,0));

  // Load the 68 point detector
  String face68 = "shape_predictor_68_face_landmarks.dat";

  // The landmark detector is implemented
  shape_predictor landmarkDetector;

  // Load the landmark model
  deserialize(face68) >> landmarkDetector;

  //Create trackbar to change lip color
  cv::createTrackbar("Hue", windowName, nullptr, 180, 0);
  cv::setTrackbarPos("Hue", windowName, 0);
  //Create trackbar to change lip color saturation
  cv::createTrackbar("Saturation", windowName, nullptr, 155, 0);
  cv::setTrackbarPos("Saturation", windowName, 155);
  //Create trackbar to change glasses color
  cv::createTrackbar("Black_White", windowName, nullptr, 1, 0);
  cv::setTrackbarPos("Black_White", windowName, 1);

  while (true)
  {
    int hueSlider = getTrackbarPos("Hue", windowName);
    int satSlider = getTrackbarPos("Saturation", windowName);
    int black_whiteSlider = getTrackbarPos("Black_White", windowName);

    // Split the image
    Mat splitImage[3];
    split(imgHSV, splitImage);

    // Set the color
    splitImage[0].setTo(Scalar(hueSlider));

    // Set the saturation
    splitImage[1].setTo(Scalar(satSlider));

    // Merge the images channels.
    merge(splitImage, 3, imgHSV);

    Mat lipImage;
    cv::cvtColor(imgHSV, lipImage, COLOR_HSV2BGR);

    // Loop over all detected face rectangles
    for (int i = 0; i < faces.size(); i++)
    {
      // For every face rectangle, run landmarkDetector
      full_object_detection landmarks = landmarkDetector(image_gray, faces[i]);

      //  Points for the glasses
      Point left_point = Point(landmarks.part(0).x(), landmarks.part(0).y());
      Point middle_point = Point(landmarks.part(27).x(), landmarks.part(16).y());
      Point right_point = Point(landmarks.part(16).x(), landmarks.part(27).y());

      // cout << left_point << ", " << middle_point << ", " << right_point << endl;

      // // Get the distance between the eye points for scaling the glasses image
      float distance_points = std::hypot(left_point.x - right_point.x, left_point.y - right_point.y);
      // cout << "Distance: " << distance_points << endl;

      // Scale the glasses image by the distance points
      float scale_factor = distance_points / image_glasses.cols;
      // cout << "Scale factor: " << scale_factor << endl;

      // resize the glasses image to fit the faces in the image
      cv::resize(image_glasses, image_glasses, cv::Size(), scale_factor, scale_factor, INTER_LINEAR);

      int width = image_glasses.size().width;
      int height = image_glasses.size().height;

      int y = middle_point.y;
      int x = middle_point.x;

      if (x % 2 == 1)
        ++x;

      if (y % 2 == 1)
        ++y;

      x = x - (width / 2);
      y = y - (height / 2);

      Mat roi;
      if (black_whiteSlider == 1)
      {
        cv::add(image(Range(y, height + y), Range(x, width + x)), image_glasses, roi);
      }
      if (black_whiteSlider == 0)
      {
        cv::subtract(image(Range(y, height + y), Range(x, width + x)), image_glasses, roi);
      }

      roi.copyTo(image(Range(y, height + y), Range(x, width + x)));

      // points for the lips
      std::vector<cv::Point> lipPoints;

      // get lip points
      for (int i = 48; i <= 67; ++i)
      {
        lipPoints.push_back(Point(landmarks.part(i).x(), landmarks.part(i).y()));
      }

      // create an area of white points on the mask
      cv::fillPoly(mask, lipPoints, Scalar(255, 255, 255), 4, 0);
    }

    // create a reverse mask
    Mat reverse_mask;
    bitwise_not(mask, reverse_mask);

    // Mask the alternated color image isolating the lips
    Mat lips;
    bitwise_and(lipImage, mask, lips);

    // Use a mask to remove the lips from the image
    Mat not_lips;
    bitwise_and(image, reverse_mask, not_lips);

    // add the lip and the no lips images together
    image = not_lips + lips;

    // Show main image
    cv::imshow(windowName, image);

    //if user press 'ESC' key
    int iKey = waitKey(1);
    if (iKey == 27)
      break;
  }

  cv::destroyWindow(windowName); // destroy the created window

  return 0;
}
