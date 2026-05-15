#pragma once

#include <opencv2/opencv.hpp>
#include "kalmanFilter.h"

using namespace cv;
using namespace std;

// Vòng đời một track trong ByteTrack.
enum TrackState { New = 0, Tracked, Lost, Removed };

// STrack là đơn vị stateful trung tâm của ByteTrack.
// Nó giữ đồng thời ba biểu diễn bbox:
// - _tlwh: quan sát gốc từ detector lúc track vừa được tạo/cập nhật
// - tlwh: trạng thái hiện tại sau Kalman, dạng [top-left x, top-left y, width, height]
// - tlbr: thuận tiện cho việc tính IoU, dạng [x1, y1, x2, y2]
class STrack
{
public:
    STrack(vector<float> tlwh_, float score);
    ~STrack();

    vector<float> static tlbr_to_tlwh(vector<float> &tlbr);
    void static multi_predict(vector<STrack*> &stracks, byte_kalman::KalmanFilter &kalman_filter);
    void static_tlwh();
    void static_tlbr();
    // xyah = [center_x, center_y, aspect_ratio, height] là hệ toạ độ Kalman của ByteTrack.
    vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);
    vector<float> to_xyah();
    void mark_lost();
    void mark_removed();
    int next_id();
    int end_frame();

    void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
    void re_activate(STrack &new_track, int frame_id, bool new_id = false);
    void update(STrack &new_track, int frame_id);

public:
    bool is_activated;
    int track_id;
    int state;

    vector<float> _tlwh;
    vector<float> tlwh;
    vector<float> tlbr;
    // frame_id: frame mới nhất track được thấy hoặc dự đoán tới
    // start_frame: frame track bắt đầu tồn tại
    // tracklet_len: số lần update liên tiếp kể từ lần kích hoạt/re-activate gần nhất
    int frame_id;
    int tracklet_len;
    int start_frame;

    // mean/covariance là state Kalman 8 chiều: [x, y, a, h, vx, vy, va, vh].
    KAL_MEAN mean;
    KAL_COVA covariance;
    float score;

private:
    byte_kalman::KalmanFilter kalman_filter;
};
