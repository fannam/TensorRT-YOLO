#ifndef UTILS_H
#define UTILS_H

#include <dirent.h>
#include <random>
#include <opencv2/opencv.hpp>

// Minimal helper for the sample CLI: list file names in a directory.
static inline int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names)
{
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

// Generates random colors for output rendering; not part of inference logic.
static inline int get_random_int(int minThres=0, int maxThres=255){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(minThres, maxThres);

    int random_integer = distrib(gen);

    return random_integer;
}

#endif  // UTILS_H
