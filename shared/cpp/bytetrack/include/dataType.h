#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

// These Eigen aliases clarify the geometric language used by ByteTrack.
// DETECTBOX represents the 4D measurement [x, y, a, h] in Kalman coordinates.
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;
typedef Eigen::Matrix<float, 1, 128, Eigen::RowMajor> FEATURE;
typedef Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor> FEATURESS;

// Kalman state:
// 8D mean = [x, y, a, h, vx, vy, va, vh]
// paired 8x8 covariance for the mean.
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

// Dynamic cost matrix for assignment/Hungarian matching.
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;
