#pragma once
#include <torch/extension.h>
#include "utils.h"

void furthest_point_sampling_cuda(int b, int n, int m, int l,
                                    const float *seeds, const float *dataset, float *temp, int *idxs);


at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples, at::Tensor seeds, bool with_seeds) {
    CHECK_CONTIGUOUS(points);
    CHECK_IS_FLOAT(points);


    at::Tensor output =
        torch::zeros({points.size(0), nsamples},
                at::device(points.device()).dtype(at::ScalarType::Int));

    at::Tensor tmp =
        torch::full({points.size(0), points.size(1)}, 1e10,
                at::device(points.device()).dtype(at::ScalarType::Float));

    if (points.is_cuda()) {
        if (with_seeds){
            CHECK_CONTIGUOUS(seeds);
            CHECK_IS_FLOAT(seeds);
            CHECK_CUDA(seeds);
            furthest_point_sampling_cuda(
                points.size(0), points.size(1), nsamples, seeds.size(1), seeds.data_ptr<float>(), points.data_ptr<float>(),
                tmp.data_ptr<float>(), output.data_ptr<int>());
        }
        else{
            furthest_point_sampling_cuda(
                points.size(0), points.size(1), nsamples, 0, seeds.data_ptr<float>(), points.data_ptr<float>(),
                tmp.data_ptr<float>(), output.data_ptr<int>());
        }
    } else {
        AT_ASSERT(false, "CPU not supported");
    }

    return output;
}
