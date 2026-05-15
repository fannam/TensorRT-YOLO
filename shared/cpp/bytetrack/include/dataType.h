#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

// Các alias Eigen ở đây làm rõ "ngôn ngữ hình học" mà ByteTrack dùng.
// DETECTBOX biểu diễn measurement 4 chiều [x, y, a, h] trong hệ toạ độ Kalman.
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;
typedef Eigen::Matrix<float, 1, 128, Eigen::RowMajor> FEATURE;
typedef Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor> FEATURESS;

// Kalman state:
// mean 8 chiều = [x, y, a, h, vx, vy, va, vh]
// covariance 8x8 đi kèm với mean.
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

using RESULT_DATA = std::pair<int, DETECTBOX>;
using TRACKER_DATA = std::pair<int, FEATURESS>;
using MATCH_DATA = std::pair<int, int>;
typedef struct t {
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
}TRACHER_MATCHD;

// Ma trận chi phí động cho assignment/Hungarian.
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;
