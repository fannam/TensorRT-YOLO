#pragma once

#include <opencv2/opencv.hpp>
#include "kalmanFilter.h"

using namespace cv;
using namespace std;

// Lifecycle of a track in ByteTrack.
enum TrackState { New = 0, Tracked, Lost, Removed };

// STrack is the core stateful unit in ByteTrack.
// It keeps three bbox representations at the same time:
// - _tlwh: the original detector observation when the track is created/updated
// - tlwh: the current post-Kalman state, in [top-left x, top-left y, width, height] format
// - tlbr: convenient for IoU computation, in [x1, y1, x2, y2] format
class STrack
{
public:
    STrack(vector<float> tlwh_, float score);
    ~STrack();

    vector<float> static tlbr_to_tlwh(vector<float> &tlbr);
    void static multi_predict(vector<STrack*> &stracks, byte_kalman::KalmanFilter &kalman_filter);
    void static_tlwh();
    void static_tlbr();
    // xyah = [center_x, center_y, aspect_ratio, height] is ByteTrack's Kalman coordinate system.
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
    // frame_id: most recent frame where the track was seen or predicted
    // start_frame: frame where the track starts to exist
    // tracklet_len: consecutive update count since the latest activate/re-activate
    int frame_id;
    int tracklet_len;
    int start_frame;

    // mean/covariance are the 8D Kalman state: [x, y, a, h, vx, vy, va, vh].
    KAL_MEAN mean;
    KAL_COVA covariance;
    float score;

private:
    byte_kalman::KalmanFilter kalman_filter;
};
