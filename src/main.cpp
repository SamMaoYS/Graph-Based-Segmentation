#include "io_image.h"
#include "seg_image.h"

struct SegmParam {
    SegmParam() : sigma(0.5), t(100), min_size(500) {}
    SegmParam(float in_sigma, float in_t, int in_min_size) : sigma(in_sigma), t(in_t), min_size(in_min_size) {}
    float sigma;
    float t;
    int min_size;
};

int main(int argc, char *argv[]) {
    std::string data_dir = "../data/rgb";
    std::vector<cv::Mat> images = io::loadMultiImages(data_dir, -1);

    std::vector<SegmParam> segm_params;
    segm_params.push_back(SegmParam(1, 90, 1000));
    segm_params.push_back(SegmParam(0.5, 200, 3000));
    segm_params.push_back(SegmParam(0.1, 500, 3000));
    segm_params.push_back(SegmParam(1.2, 500, 3000));
    segm_params.push_back(SegmParam(0.2, 100, 3000));
    segm_params.push_back(SegmParam(0.4, 100, 3000));
    for (int i = 0; i < images.size(); ++i) {
        cv::Mat image = images[i];
        cv::Mat result;
        std::vector<cv::Mat> masks;
        Segmentation *seg = new Segmentation(image);
        seg->setParameters(segm_params[i].sigma, segm_params[i].t, segm_params[i].min_size);
        seg->run();
        seg->getResult(result);
        seg->getSegmentMasks(masks);

        if (!result.empty()) {
            std::cout << "Segment numbers " << seg->getSegmentNumber() << std::endl;
            cv::namedWindow("Display", cv::WINDOW_AUTOSIZE);
            cv::imshow("Display", result);
            cv::imwrite("../data/result/seg" + std::to_string(i) + ".png", result);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

//        for (int j = 0; j < masks.size(); ++j) {
//            cv::Mat mask_out;
//            mask_out = masks[j].mul(255);
//            cv::namedWindow("Masks", cv::WINDOW_AUTOSIZE);
//            cv::imshow("Masks", mask_out);
//            cv::imwrite("../../data/result/masks/seg" + std::to_string(i) + "_mask" + std::to_string(j) + ".png", mask_out);
//            cv::waitKey(0);
//            cv::destroyAllWindows();
//        }

        image.release();
        result.release();
        delete seg;
    }

    return 0;
}