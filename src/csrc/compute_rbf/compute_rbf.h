#pragma once
#include <torch/extension.h>
#include "utils.h"

void compute_rbf_cuda(int batch_size, int query_size, int kernel_size,
                    const float *points, const float *centers, const float *rotates, const float *scales, float *output);


at::Tensor compute_rbf(at::Tensor points, at::Tensor centers, at::Tensor rotates, at::Tensor scales) {
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(centers);
    CHECK_CONTIGUOUS(rotates);
    CHECK_CONTIGUOUS(scales);
    CHECK_IS_FLOAT(points);
    CHECK_IS_FLOAT(centers);
    CHECK_IS_FLOAT(rotates);
    CHECK_IS_FLOAT(scales);

    at::Tensor output =
        torch::zeros({points.size(0), points.size(1), centers.size(1)},
                at::device(points.device()).dtype(at::ScalarType::Float));

    if (points.is_cuda()) {
        compute_rbf_cuda(
            points.size(0), points.size(1), centers.size(1), 
            points.data_ptr<float>(), centers.data_ptr<float>(),
            rotates.data_ptr<float>(), scales.data_ptr<float>(), 
            output.data_ptr<float>());
    } else {
        AT_ASSERT(false, "CPU not supported");
    }

    return output;
}
