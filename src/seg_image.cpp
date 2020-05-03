//
// Created by sam on 2020-05-03.
//

#include "seg_image.h"
#include <opencv2/imgproc.hpp>

using namespace std;

Segmentation::Segmentation() {
    sigma = 0.5;
    t = 500;
}

Segmentation::Segmentation(const cv::Mat &image_in) {
    if (!images.empty())
        images.clear();
    images.push_back(image_in);
    sigma = 0.5;
    t = 500;
}

Segmentation::Segmentation(const std::vector<cv::Mat> &images_in) {
    if (!images.empty())
        images.clear();
    images = images_in;
    sigma = 0.5;
    t = 500;
}

Segmentation::~Segmentation() {
    if (!images.empty())
        images.clear();

    if (!channels.empty()) {
        for (size_t i = 0; i < channels.size(); ++i)
            channels[i].clear();
        channels.clear();
    }

    if (!results.empty())
        results.clear();
}

void Segmentation::setParameters(const float sigma_in, const float t_in) {
    sigma = sigma_in;
    t = t_in;
}

void Segmentation::run() {
    for (cv::Mat image : images) {
        cv::Mat tmp_result;

        filter(image, tmp_result);
        results.push_back(tmp_result);
    }
}

void Segmentation::filter(const cv::Mat &image_in, cv::Mat &image_out) {
    uint8_t len = ceil(sigma * 4.0f) + 1;
    len = (len % 2 == 0) ? len + 1 : len;
    cv::Size k_size(len, len);
    cv::GaussianBlur(image_in, image_out, k_size, sigma);
}