#include "io_image.h"
#include "seg_image.h"

int main(int argc, char *argv[]) {
    std::string data_dir = "../../data";
//    vector<string> suffixes = {".png"};
    std::vector<cv::Mat> images = io::loadMultiImages(data_dir, -1);

    std::vector<cv::Mat> results;
    Segmentation *seg = new Segmentation(images);
    seg->setParameters(0.5, 500);
    seg->run();
    seg->getResults(results);

    if (!results.empty()) {
        cv::namedWindow("Display", cv::WINDOW_KEEPRATIO);
        cv::imshow("Display", results[0]);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    images.clear();
    results.clear();

    return 0;
}