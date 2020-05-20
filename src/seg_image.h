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

    ~Segmentation();

    void setParameters(const float sigma_in, const float t_in, const float min_size);

    void run();

    inline void getResult(cv::Mat &buffer_in) { buffer_in = result_.clone(); }

    inline void getSegmentMasks(std::vector<cv::Mat> &buffer_in) { buffer_in = segments_; }

    inline int getSegmentNumber() { return num_segs_; }

protected:
    struct Edge {
        float w;
        int a, b;
    };

    struct Element {
        int rank;
        int p;
        int size;
    };

    struct Universe {
        Universe() {}

        Universe(int num_elements) : num(num_elements) {
            elements = new Element[num_elements];
            for (size_t i = 0; i < num_elements; i++) {
                elements[i].rank = 0;
                elements[i].p = i;
                elements[i].size = 1;
            }
        }

        ~Universe() {
            delete[] elements;
        }

        int find(int x) {
            int y = x;
            while (y != elements[y].p)
                y = elements[y].p;
            elements[x].p = y;
            return y;
        }

        void join(int x, int y) {
            if (elements[x].rank > elements[y].rank) {
                elements[y].p = x;
                elements[x].size += elements[y].size;
            } else {
                elements[x].p = y;
                elements[y].size += elements[x].size;
                if (elements[x].rank == elements[y].rank)
                    elements[y].rank++;
            }
            this->num--;
        }

        Element *elements;
        int num;
    };

    void filter(const cv::Mat &image_in, cv::Mat &image_out);

    float pixelDiff(const cv::Mat &r, const cv::Mat &g, const cv::Mat &b, const int x1, const int y1, const int x2,
                    const int y2);

    int buildGraph(const cv::Mat &image_in);

    cv::Mat segment(const int width, const int height, const int num_edges);

    void randomColor(cv::Vec3b &color);

private:
    cv::Mat image_;
    cv::Mat result_;
    std::vector<cv::Mat> segments_;
    Edge *graph_;
    Universe *universe_;
    int num_segs_;
    float sigma_;
    float t_;
    float min_size_;
};

#endif //SEGMENTATION_SEG_IMAGE_H
