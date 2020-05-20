//
// Created by sam on 2020-05-03.
//

#include "seg_image.h"
#include <map>
#include <opencv2/imgproc.hpp>

using namespace std;

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
    filter(image_, result_);

    int num_edges = buildGraph(result_);
    result_ = segment(image_.cols, image_.rows, num_edges);
}

void Segmentation::filter(const cv::Mat &image_in, cv::Mat &image_out) {
    uint8_t len = ceil(sigma_ * 4.0f) + 1;
    len = (len % 2 == 0) ? len + 1 : len;
    cv::Size k_size(len, len);
    cv::GaussianBlur(image_in, image_out, k_size, sigma_);
}

float
Segmentation::pixelDiff(const cv::Mat &r, const cv::Mat &g, const cv::Mat &b, const int x1, const int y1, const int x2,
                        const int y2) {
    float diff_r = r.at<float>(y1, x1) - r.at<float>(y2, x2);
    float diff_g = g.at<float>(y1, x1) - g.at<float>(y2, x2);
    float diff_b = b.at<float>(y1, x1) - b.at<float>(y2, x2);

    return sqrt(diff_r * diff_r + diff_g * diff_g + diff_b * diff_b);
}

int Segmentation::buildGraph(const cv::Mat &image_in) {
    vector<cv::Mat> bgr(3);
    cv::split(image_in, bgr);

    cv::Mat r, g, b;
    bgr[2].convertTo(r, CV_32F);
    bgr[1].convertTo(g, CV_32F);
    bgr[0].convertTo(b, CV_32F);

    int width = image_in.cols;
    int height = image_in.rows;

    graph_ = new Edge[width * height * 4];
    int num = 0;
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            if (x < width - 1) {
                graph_[num].a = y * width + x;
                graph_[num].b = y * width + (x + 1);
                graph_[num].w = pixelDiff(r, g, b, x, y, x + 1, y);
                num++;
            }

            if (y < height - 1) {
                graph_[num].a = y * width + x;
                graph_[num].b = (y + 1) * width + x;
                graph_[num].w = pixelDiff(r, g, b, x, y, x, y + 1);
                num++;
            }

            if ((x < width - 1) && (y < height - 1)) {
                graph_[num].a = y * width + x;
                graph_[num].b = (y + 1) * width + (x + 1);
                graph_[num].w = pixelDiff(r, g, b, x, y, x + 1, y + 1);
                num++;
            }

            if ((x < width - 1) && (y > 0)) {
                graph_[num].a = y * width + x;
                graph_[num].b = (y - 1) * width + (x + 1);
                graph_[num].w = pixelDiff(r, g, b, x, y, x + 1, y - 1);
                num++;
            }
        }
    }

    bgr.clear();
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
