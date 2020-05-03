//
// Created by sam on 2020-05-03.
//

#ifndef SEGMENTATION_SEG_IMAGE_H
#define SEGMENTATION_SEG_IMAGE_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

class Segmentation {
public:
    Segmentation();

    explicit Segmentation(const cv::Mat &image_in);

    explicit Segmentation(const std::vector<cv::Mat> &images_in);

    ~Segmentation();

    void setParameters(const float sigma_in, const float t_in);

    void run();

    inline void getResults(std::vector<cv::Mat> &buffer_in) { buffer_in = results; }

    inline void getResult(cv::Mat &buffer_in) { buffer_in = results[0]; }

protected:
    void filter(const cv::Mat &image_in, cv::Mat &image_out);

private:
    std::vector<cv::Mat> images;
    std::vector<std::vector<cv::Mat>> channels;
    std::vector<cv::Mat> results;
    float sigma;
    float t;
};


#endif //SEGMENTATION_SEG_IMAGE_H
