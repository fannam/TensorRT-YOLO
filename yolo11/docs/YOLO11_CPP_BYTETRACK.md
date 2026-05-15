# YOLO11 C++ ByteTrack Appendix

This document focuses only on `shared/cpp/bytetrack` and how `yolo11/cpp/track` uses it.

## 1. Core Abstractions

- `Object`
  Tracker input detection: `rect + label + prob`
- `STrack`
  A track with state and history
- `TrackState`
  `New`, `Tracked`, `Lost`, `Removed`
- `BYTETracker`
  Coordinates the full per-frame update pipeline
- `KalmanFilter`
  Predicts and updates box motion state
- `lapjv`
  Assignment solver for the cost matrix

## 2. The Three Bounding-Box Coordinate Systems

- `tlwh`
  `[top-left x, top-left y, width, height]`
- `tlbr`
  `[x1, y1, x2, y2]`
- `xyah`
  `[center_x, center_y, aspect_ratio, height]`

ByteTrack uses:

- `tlwh` for detector-adjacent communication and drawing
- `tlbr` for IoU computation
- `xyah` as the Kalman measurement/state representation

## 3. Kalman State Vector

8D state:

`[x, y, a, h, vx, vy, va, vh]`

Where:

- `x, y`
  bbox center
- `a`
  aspect ratio = `w / h`
- `h`
  bbox height
- `vx, vy, va, vh`
  corresponding velocities

`xyah` is used instead of `xywh` because aspect ratio is more stable than absolute width in many tracking scenes.

## 4. ByteTrack State-Machine Update

You can read `BYTETracker::update()` as five stages:

### Step 1. Split high-score and low-score detections

- `score >= track_thresh`
  goes into `detections`
- everything else
  goes into `detections_low`

The core ByteTrack idea is to avoid discarding low-score detections entirely.

### Step 2. First association with `tracked + lost`

- merge `tracked_stracks` and `lost_stracks` into `strack_pool`
- run `STrack::multi_predict()` so Kalman predicts positions for the current frame
- compute `iou_distance()`
- solve assignment with `lapjv()`

Result:

- matched tracks are `update()`d or `re_activate()`d
- unmatched tracks continue to the next stage

### Step 3. Second association with low-score detections

Only tracks currently in `Tracked` state that missed the first pass participate here.

Purpose:

- preserve continuity when the detector gives an object a low score for a few frames
- reduce ID switches

### Step 4. Handle unconfirmed and new tracks

- `unconfirmed`
  means tracks that have appeared for too few frames
- if an unconfirmed track cannot be matched again
  -> `Removed`
- leftover detections that still meet `high_thresh`
  -> `activate()` as new tracks

### Step 5. Clean up state pools

- tracks lost for longer than `max_time_lost`
  -> `Removed`
- merge `tracked`, `lost`, and `removed`
- remove duplicate tracks with `remove_duplicate_stracks()`

## 5. How Kalman Predict/Update Works

### `initiate()`

Create a new track:

- convert the detector measurement to `xyah`
- initialize velocity to zero
- initialize covariance based on bbox size

### `predict()`

Advance the state to the next frame:

- `x += vx`
- `y += vy`
- `a += va`
- `h += vh`

and add motion noise.

### `project()`

Project the 8D state into 4D measurement space for comparison with a new detection.

### `update()`

Use the detector measurement to correct the prediction:

- innovation = detection - prediction
- Kalman gain decides how much to trust the detector

## 6. IoU and Assignment Flow

`iou_distance()` does:

1. convert tracks to `tlbr`
2. compute the IoU matrix
3. convert it to a cost matrix using `1 - IoU`

`linear_assignment()` does:

1. call `lapjv()`
2. return:
   - `matches`
   - `unmatched_a`
   - `unmatched_b`

`cost_limit` is the threshold that determines whether a pair is still accepted.

## 7. Meaning of Key `STrack` Fields

- `is_activated`
  the track is stable enough to be emitted
- `tracklet_len`
  number of consecutive frames the track has been updated since the last activate/re-activate
- `frame_id`
  the last frame in which the track was seen or predicted
- `start_frame`
  the frame where the track was created
- `track_id`
  the stable ID used for drawing on output video

## 8. How `track/main.cpp` Connects Detector and Tracker

1. run `YoloDetector::inference(img)`
2. filter classes using `trackClasses`
3. convert bbox from `xyxy` to `tlwh`
4. create `Object`
5. call `tracker.update(objects)`
6. draw `track_id` and the tracked bbox

Important points:

- the tracker consumes detections already scaled to the original image
- the tracker knows nothing about TensorRT, CUDA, or the YOLO head layout
- the only coupling between detector and tracker is `bbox + score + label`
