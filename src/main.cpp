#include "io_image.h"
#include "seg_image.h"

int main(int argc, char *argv[]) {
    std::string data_dir = "../../data/rgb";
    std::vector<cv::Mat> images = io::loadMultiImages(data_dir, -1);

    for (int i = 0; i < images.size(); ++i) {
        cv::Mat image = images[i];
        cv::Mat result;
        std::vector<cv::Mat> masks;
        Segmentation *seg = new Segmentation(image);
        if (i==0)
            seg->setParameters(0.5, 3000, 1000);
        else if (i == 1)
            seg->setParameters(0.5, 1000, 500);
        else if (i == 2)
            seg->setParameters(1, 3000, 1000);
        seg->run();
        seg->getResult(result);
        seg->getSegmentMasks(masks);

        if (!result.empty()) {
            std::cout << "Segment numbers " << seg->getSegmentNumber() << std::endl;
            cv::namedWindow("Display", cv::WINDOW_AUTOSIZE);
            cv::imshow("Display", result);
            cv::imwrite("../../data/result/seg" + std::to_string(i) + ".png", result);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

        for (int j = 0; j < masks.size(); ++j) {
            cv::Mat mask_out;
            mask_out = masks[j].mul(255);
            cv::namedWindow("Masks", cv::WINDOW_AUTOSIZE);
            cv::imshow("Masks", mask_out);
            cv::imwrite("../../data/result/masks/seg" + std::to_string(i) + "_mask" + std::to_string(j) + ".png", mask_out);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

        image.release();
        result.release();
        delete seg;
    }

    return 0;
}