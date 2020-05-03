#include <iostream>
#include <experimental/filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
namespace fs = experimental::filesystem;

cv::Mat loadImage(const string &dir, const int flag);
vector<cv::Mat> loadMultiImages(const string &dir, const int flag, const vector<string> &suffixes = {});

// return number -1 means dir doesn't exist, 0 means no file found, 1 means successfully find the files
int getDataFiles(const string &dir, vector<string> &file_paths, const vector<string> &suffixes = {});

bool pathExists(const fs::path &p, fs::file_status s = fs::file_status{});

bool checkSuffix(const string &suffix, const vector<string> &suffixes);

int main(int argc, char *argv[]) {
    string data_dir = "../../data";
//    vector<string> suffixes = {".png"};
    vector<cv::Mat> images = loadMultiImages(data_dir, -1);

    if (!images.empty()) {
        cv::namedWindow("Display", cv::WINDOW_KEEPRATIO);
        cv::imshow("Display", images[0]);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return 0;
}

cv::Mat loadImage(const string &dir, const int flag) {
    fs::path path(dir);
    cv::Mat image;

    if (path.extension() == "") {
        cout << "Input directory is to a folder, maybe you want to use loadMultiImages to load multiple images from a folder" << endl;
    }
    else {
        image = cv::imread(dir, flag);
        if (!image.data) {
            fs::path path(dir);
            cerr << "Could not load image " << path.filename().string() << endl;
        }
    }

    return image;
}
vector<cv::Mat> loadMultiImages(const string &dir, const int flag, const vector<string> &suffixes) {
    vector<string> file_paths;
    vector<cv::Mat> images;
    int status = getDataFiles(dir, file_paths, suffixes);
    if (status == 1) {
        for (size_t i = 0; i < file_paths.size(); ++i) {
            cv::Mat image = cv::imread(file_paths[i], flag);
            if (!image.data) {
                fs::path path(file_paths[i]);
                cerr << "Could not load image " << path.filename().string() << endl;
                continue;
            }
            images.push_back(image);
        }
    }
    else if (status == 0){
        cout << "Could not find file with the specific suffixes" << endl;
    }
    else{
        cout << "The directory doesn't exist" << endl;
    }

    return images;
}

int getDataFiles(const string &dir, vector<string> &file_paths, const vector<string> &suffixes) {
    fs::path path(dir);
    if (!pathExists(path)) {
        return -1;
    }

    string file_name = path.filename().string();
    // passed in dir is a dir to folder
    if (path.extension() == "") {
        string file_dir = path.string();
        if (file_name==".")
            file_dir = path.parent_path().string();
        for (auto iter = fs::directory_iterator(file_dir); iter != fs::directory_iterator(); ++iter) {
            if (!pathExists(*iter, iter->status()))
                continue;
            fs::path tmp_path(*iter);
            string tmp_suffix = tmp_path.extension().string();
           if (checkSuffix(tmp_suffix, suffixes))
                file_paths.push_back(tmp_path.string());
        }
    }
        // passed in dir is a dir to file
    else {
        if (!file_paths.empty())
            file_paths.clear();

        string tmp_suffix = path.extension().string();
        if (checkSuffix(tmp_suffix, suffixes))
            file_paths.push_back(dir);
    }

    return !file_paths.empty();
}

bool pathExists(const fs::path &p, fs::file_status s) {
    return fs::status_known(s) ? fs::exists(s) : fs::exists(p);
}

bool checkSuffix(const string &suffix, const vector<string> &suffixes) {
    auto iter = find(suffixes.begin(), suffixes.end(), suffix);
    return (iter != suffixes.end() || suffixes.empty());
}