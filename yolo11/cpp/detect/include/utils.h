#ifndef UTILS_H
#define UTILS_H

#include <dirent.h>
#include <random>
#include <opencv2/opencv.hpp>

// Helper tối giản cho sample CLI: liệt kê tên file trong một thư mục.
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

// Sinh màu ngẫu nhiên cho bước vẽ output; không tham gia logic suy luận.
static inline int get_random_int(int minThres=0, int maxThres=255){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(minThres, maxThres);

    int random_integer = distrib(gen);

    return random_integer;
}

#endif  // UTILS_H
