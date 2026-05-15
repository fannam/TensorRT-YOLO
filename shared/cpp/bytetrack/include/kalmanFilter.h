#pragma once

#include "dataType.h"

namespace byte_kalman
{
    // KalmanFilter của ByteTrack vận hành trong không gian xyah:
    // x, y là tâm box; a là aspect ratio; h là chiều cao.
    class KalmanFilter
    {
    public:
        static const double chi2inv95[10];
        KalmanFilter();
        KAL_DATA initiate(const DETECTBOX& measurement);
        void predict(KAL_MEAN& mean, KAL_COVA& covariance);
        KAL_HDATA project(const KAL_MEAN& mean, const KAL_COVA& covariance);
        KAL_DATA update(const KAL_MEAN& mean,
            const KAL_COVA& covariance,
            const DETECTBOX& measurement);

        Eigen::Matrix<float, 1, -1> gating_distance(
            const KAL_MEAN& mean,
            const KAL_COVA& covariance,
            const std::vector<DETECTBOX>& measurements,
            bool only_position = false);

    private:
        // motion_mat: từ state hiện tại sang state frame kế tiếp
        // update_mat: chiếu state 8D xuống measurement 4D
        Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
        Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
        float _std_weight_position;
        float _std_weight_velocity;
    };
}
