//
// Created by sam on 2020-05-03.
//

#include "seg_image.h"
#include <map>
#include <opencv2/imgproc.hpp>
#include <opencv2/saliency.hpp>

using namespace std;

#ifndef rad2deg
#define rad2deg 180.0f/M_PI
#endif

#ifndef deg2rad
#define deg2rad M_PI/180.0f
#endif

#define THRESHOLD(size, c) (c/size)

Segmentation::Segmentation() {
    sigma_ = 0.5;
    t_ = 500;
    min_size_ = 100;
}

Segmentation::Segmentation(const cv::Mat &image_in) {
    if (!image_.empty())
        image_.release();
    image_ = image_in.clone();
    sigma_ = 0.5;
    t_ = 500;
    min_size_ = 100;
}

Segmentation::~Segmentation() {
    if (!image_.empty())
        image_.release();

    if (!result_.empty())
        result_.release();

    delete graph_;
    delete universe_;
}

void Segmentation::setParameters(const float sigma_in, const float t_in, const float min_size) {
    sigma_ = sigma_in;
    t_ = t_in;
    min_size_ = min_size;
}

void Segmentation::run() {
    cv::Mat gray, smoothed;
    cv::cvtColor(image_, gray, cv::COLOR_BGR2GRAY);
    filter(gray, smoothed);

	cv::saliency::StaticSaliencyFineGrained sal_generator = cv::saliency::StaticSaliencyFineGrained();
	sal_generator.computeSaliency(gray, saliency_);
    saliency_.convertTo(saliency_, CV_32F);
	cv::normalize(saliency_, saliency_, 1.0, 0.0, cv::NORM_MINMAX);
	cv::pow(saliency_, 2, saliency_);
//    cv::imshow("saliency", saliency_);
//    cv::waitKey(0);
//    cv::destroyAllWindows();

    cv::Canny(gray, edge_, 100, 200);
    cv::GaussianBlur(edge_, edge_, cv::Size(5, 5), 0);
    edge_.convertTo(edge_, CV_32F);
    cv::normalize(edge_, edge_, 1.0, 0.0, cv::NORM_MINMAX);
//    cv::imshow("edge", edge_);
//    cv::waitKey(0);
//    cv::destroyAllWindows();

    int num_edges = buildGraph(image_);
    result_ = segment(image_.cols, image_.rows, num_edges);
}

void Segmentation::filter(const cv::Mat &image_in, cv::Mat &image_out) {
    uint8_t len = ceil(sigma_ * 4.0f) + 1;
    len = (len % 2 == 0) ? len + 1 : len;
    cv::Size k_size(len, len);
    cv::GaussianBlur(image_in, image_out, k_size, sigma_);
}

float
Segmentation::pixelDiff(const cv::Mat &h, const cv::Mat &s, const cv::Mat &v, const int x1, const int y1, const int x2,
                        const int y2) {
    float k_dv = 4.5f;
    float k_ds = 0.1f;

    float h1 = h.at<float>(y1, x1);
    float s1 = s.at<float>(y1, x1);
    float v1 = v.at<float>(y1, x1);

    float h2 = h.at<float>(y2, x2);
    float s2 = s.at<float>(y2, x2);
    float v2 = v.at<float>(y2, x2);

    float delta_v = k_dv * fabs(v1 - v2);
    float delta_h = fabs(h1 - h2);
    float theta = 0.0f;

    if (delta_h < 180)
        theta = delta_h;
    else
        theta = 360 - delta_h;

    float delta_s = k_ds * sqrt(s1 * s1 + s2 * s2 - 2 * s1 * s2 * cos(theta * deg2rad));
    float delta_hsv = sqrt(delta_v * delta_v + delta_s * delta_s);
    if (isnan(delta_hsv) || isinf(delta_hsv))
        delta_hsv = 0.0f;

    if (v1 < 0.03f)
        delta_hsv = 0.01f;

    delta_hsv /= (sqrtf(k_dv * k_dv + k_ds * k_ds));
    delta_hsv = std::log2(1+delta_hsv);

    float delta_sal = saliency_.at<float>(y1, x1) - saliency_.at<float>(y2, x2);
    delta_sal = (delta_sal < 0)? 0.0 : delta_sal;
    delta_sal = std::log2(1+delta_sal) * 2;

    float delta_edge = edge_.at<float>(y1, x1) - edge_.at<float>(y2, x2);
    delta_edge = (delta_edge < 0)? 0.0 : delta_edge;
    delta_edge = std::log2(1+delta_edge) * 1;

    return delta_hsv+delta_sal+delta_edge;
}

int Segmentation::buildGraph(const cv::Mat &image_in) {
    cv::Mat hsv_img;
    cv::cvtColor(image_in, hsv_img, cv::COLOR_BGR2HSV);
    filter(hsv_img, hsv_img);
    vector<cv::Mat> hsv(3);
    cv::split(hsv_img, hsv);

    cv::Mat h, s, v;
    hsv[2].convertTo(h, CV_32F);
    hsv[1].convertTo(s, CV_32F);
    hsv[0].convertTo(v, CV_32F);

    int width = image_in.cols;
    int height = image_in.rows;

    graph_ = new Edge[width * height * 4];
    int num = 0;
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            if (x < width - 1) {
                graph_[num].a = y * width + x;
                graph_[num].b = y * width + (x + 1);
                graph_[num].w = pixelDiff(h, s, v, x, y, x + 1, y);
                num++;
            }

            if (y < height - 1) {
                graph_[num].a = y * width + x;
                graph_[num].b = (y + 1) * width + x;
                graph_[num].w = pixelDiff(h, s, v, x, y, x, y + 1);
                num++;
            }

            if ((x < width - 1) && (y < height - 1)) {
                graph_[num].a = y * width + x;
                graph_[num].b = (y + 1) * width + (x + 1);
                graph_[num].w = pixelDiff(h, s, v, x, y, x + 1, y + 1);
                num++;
            }

            if ((x < width - 1) && (y > 0)) {
                graph_[num].a = y * width + x;
                graph_[num].b = (y - 1) * width + (x + 1);
                graph_[num].w = pixelDiff(h, s, v, x, y, x + 1, y - 1);
                num++;
            }
        }
    }

    hsv.clear();
    return num;
}

cv::Mat Segmentation::segment(const int width, const int height, const int num_edges) {
    int num_vertices = width * height;
    sort(graph_, graph_ + num_edges, [](const Edge &a, const Edge &b) { return a.w < b.w; });

    universe_ = new Universe(num_vertices);

    float *thresholds = new float[num_vertices];
    for (size_t i = 0; i < num_vertices; i++)
        thresholds[i] = THRESHOLD(1, t_);

    for (size_t i = 0; i < num_edges; i++) {
        Edge e = graph_[i];
        int a = universe_->find(e.a);
        int b = universe_->find(e.b);
        if (a != b) {
            if ((e.w <= thresholds[a]) && (e.w <= thresholds[b])) {
                universe_->join(a, b);
                a = universe_->find(a);
                thresholds[a] = e.w + THRESHOLD(universe_->elements[a].size, t_);
            }
        }
    }

    for (size_t i = 0; i < num_edges; i++) {
        Edge e = graph_[i];
        int a = universe_->find(e.a);
        int b = universe_->find(e.b);
        if ((a != b) && ((universe_->elements[a].size < min_size_) || (universe_->elements[b].size < min_size_))) {
            universe_->join(a, b);
        }
    }

    num_segs_ = universe_->num;

    cv::Vec3b *colors = new cv::Vec3b[num_vertices];
    for (size_t i = 0; i < num_vertices; i++)
        randomColor(colors[i]);

    cv::Mat seg_img(height, width, CV_8UC3, cv::Scalar::all(0));
    for (int i = 0; i < num_segs_; ++i)
        segments_.push_back(cv::Mat(height, width, CV_8U, cv::Scalar::all(0)));

    map<int, int> color_idx_map;
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            int c = universe_->find(y * width + x);
            if (color_idx_map.find(c) == color_idx_map.end())
                color_idx_map[c] = color_idx_map.size();
            seg_img.at<cv::Vec3b>(y, x) = colors[c];
            segments_[color_idx_map[c]].at<uchar>(y,x) = 1;
        }
    }

    delete thresholds;
    delete colors;

    return seg_img;
}

void Segmentation::randomColor(cv::Vec3b &color) {
    color[0] = (uchar) rand();
    color[1] = (uchar) rand();
    color[2] = (uchar) rand();
}
