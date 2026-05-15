#pragma once

#include "STrack.h"

// Object là "ngôn ngữ chung" giữa detector và ByteTrack:
// detector chỉ cần đưa bbox tlwh, class label và score vào tracker.
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

// BYTETracker giữ ba pool track chính:
// - tracked_stracks: đang hoạt động
// - lost_stracks: tạm mất nhưng còn cơ hội hồi sinh
// - removed_stracks: quá hạn hoặc bị loại hẳn
class BYTETracker
{
public:
    BYTETracker(int frame_rate = 30, int track_buffer = 30);
    ~BYTETracker();

    // update() nhận detection của một frame và trả về các track còn active sau khi:
    // tách high/low score, ghép IoU hai lượt, xử lý unconfirmed và loại track quá hạn.
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
    // Các ngưỡng chính của ByteTrack:
    // track_thresh tách high/low score, high_thresh quyết định khởi tạo track mới,
    // match_thresh là ngưỡng cost tối đa ở lượt ghép đầu.
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
