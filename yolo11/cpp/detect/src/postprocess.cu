#include "postprocess.h"

// This file implements GPU postprocess for detect:
// layout transpose, class/bbox decode, and non-maximum suppression.

// ------------------ transpose --------------------
__global__ void transpose_kernel(float* src, float* dst, int numBboxes, int numElements, int edge){
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    // src is [numElements, numBboxes]; dst becomes [numBboxes, numElements].
    dst[position] = src[(position % numElements) * numBboxes + position / numElements];
}

void transpose(float* src, float* dst, int numBboxes, int numElements, cudaStream_t stream){
    int edge = numBboxes * numElements;
    int blockSize = 256;
    int gridSize = (edge + blockSize - 1) / blockSize;
    transpose_kernel<<<gridSize, blockSize, 0, stream>>>(src, dst, numBboxes, numElements, edge);
}

// ------------------ decode ( get class and conf ) --------------------
__global__ void decode_kernel(float* src, float* dst, int numBboxes, int numClasses, float confThresh, int maxObjects, int numBoxElement){
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= numBboxes) return;

    // After transpose, each candidate is [cx, cy, w, h, cls0, cls1, ...].
    float* pitem = src + (4 + numClasses) * position;
    float* classConf = pitem + 4;
    float confidence = 0;
    int label = 0;
    for (int i = 0; i < numClasses; i++){
        if (classConf[i] > confidence){
            confidence = classConf[i];
            label = i;
        }
    }

    if (confidence < confThresh) return;

    // dst[0] stores the valid box count. atomicAdd allows multiple threads to append boxes concurrently.
    int index = (int)atomicAdd(dst, 1);
    if (index >= maxObjects) return;

    float cx     = pitem[0];
    float cy     = pitem[1];
    float width  = pitem[2];
    float height = pitem[3];

    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    float* pout_item = dst + 1 + index * numBoxElement;
    pout_item[0] = left;
    pout_item[1] = top;
    pout_item[2] = right;
    pout_item[3] = bottom;
    pout_item[4] = confidence;
    pout_item[5] = label;
    // keep_flag is set to 0 by NMS if the box is suppressed.
    pout_item[6] = 1;
}

void decode(float* src, float* dst, int numBboxes, int numClasses, float confThresh, int maxObjects, int numBoxElement, cudaStream_t stream){
    // Only the first counter element needs to be reset; later box slots will be overwritten as valid boxes appear.
    cudaMemsetAsync(dst, 0, sizeof(int), stream);
    int blockSize = 256;
    int gridSize = (numBboxes + blockSize - 1) / blockSize;
    decode_kernel<<<gridSize, blockSize, 0, stream>>>(src, dst, numBboxes, numClasses, confThresh, maxObjects, numBoxElement);
}

// ------------------ nms --------------------
__device__ float box_iou(
    float aleft, float atop, float aright, float abottom,
    float bleft, float btop, float bright, float bbottom
){
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f) return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

__global__ void nms_kernel(float* data, float kNmsThresh, int maxObjects, int numBoxElement){
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    int count = min((int)data[0], maxObjects);
    if (position >= count) return;

    // Layout of each box: [x1, y1, x2, y2, conf, class_id, keep_flag].
    float* pcurrent = data + 1 + position * numBoxElement;
    float* pitem;
    for (int i = 0; i < count; i++){
        pitem = data + 1 + i * numBoxElement;
        if (i == position || pcurrent[5] != pitem[5]) continue;

        // Tie-break rule: keep the higher-score box; if scores match, prefer the earlier box.
        if (pitem[4] >= pcurrent[4]){
            if (pitem[4] == pcurrent[4] && i < position) continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0], pitem[1], pitem[2], pitem[3]
            );

            if (iou > kNmsThresh){
                pcurrent[6] = 0;
                return;
            }
        }
    }
}

void nms(float* data, float kNmsThresh, int maxObjects, int numBoxElement, cudaStream_t stream){
    // Using maxObjects instead of the real count keeps the grid shape fixed; the kernel exits once position >= count.
    int blockSize = maxObjects < 256?maxObjects:256;
    int gridSize = (maxObjects + blockSize - 1) / blockSize;
    nms_kernel<<<gridSize, blockSize, 0, stream>>>(data, kNmsThresh, maxObjects, numBoxElement);
}
