#pragma once

#include "STrack.h"

// Object is the shared language between the detector and ByteTrack:
// the detector only needs to pass tlwh bbox, class label, and score into the tracker.
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

// BYTETracker keeps three main track pools:
// - tracked_stracks: currently active
// - lost_stracks: temporarily missing but still recoverable
// - removed_stracks: expired or permanently removed
class BYTETracker
{
public:
    BYTETracker(int frame_rate = 30, int track_buffer = 30);
    ~BYTETracker();

    // update() receives one frame's detections and returns the tracks still active after:
    // splitting high/low scores, running two IoU association passes, handling unconfirmed tracks, and dropping expired tracks.
    vector<STrack> update(const vector<Object>& objects);
    Scalar get_color(int idx);

private:
    vector<STrack*> joint_stracks(vector<STrack*> &tlista, vector<STrack> &tlistb);
    vector<STrack> joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);

    vector<STrack> sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);
    void remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb);

    void linear_assignment(vector<vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
        vector<vector<int> > &matches, vector<int> &unmatched_a, vector<int> &unmatched_b);
    vector<vector<float> > iou_distance(vector<STrack*> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size);
    vector<vector<float> > iou_distance(vector<STrack> &atracks, vector<STrack> &btracks);
    vector<vector<float> > ious(vector<vector<float> > &atlbrs, vector<vector<float> > &btlbrs);

    double lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol,
        bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:
    // Main ByteTrack thresholds:
    // track_thresh splits high/low scores, high_thresh decides whether to start a new track,
    // and match_thresh is the maximum cost allowed in the first association pass.
    float track_thresh;
    float high_thresh;
    float match_thresh;
    int frame_id;
    int max_time_lost;

    vector<STrack> tracked_stracks;
    vector<STrack> lost_stracks;
    vector<STrack> removed_stracks;
    byte_kalman::KalmanFilter kalman_filter;
};
