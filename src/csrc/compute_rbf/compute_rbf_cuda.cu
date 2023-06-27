#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// Input points: (b, nq, 3), centers: (b, nk, 3), rotates: (b, nk, 9), scales: (b, nk, 3)
// Ouput (b, nq, nk)
__global__ void compute_rbf_kernel(
  int batch_size, int query_size, int kernel_size,
  const float *points, const float *centers, const float *rotates, const float *scales, float *output) {

  int BLOCK_SIZE = kernel_size;
  extern __shared__ float shared_block[];
  
  int batch_id = blockIdx.x;
  int block_id = blockIdx.y;
  int thread_id = threadIdx.x;
  int query_id_start = batch_id * query_size + block_id * BLOCK_SIZE;
  int output_id_start = batch_id * query_size * kernel_size + block_id * BLOCK_SIZE * kernel_size;

  if (query_id_start + thread_id < batch_id * query_size + query_size){
    shared_block[thread_id * 3 + 0] = points[(query_id_start + thread_id) * 3 + 0];
    shared_block[thread_id * 3 + 1] = points[(query_id_start + thread_id) * 3 + 1];
    shared_block[thread_id * 3 + 2] = points[(query_id_start + thread_id) * 3 + 2];
  }
  __syncthreads();

  float kernel_centers[3];
  float kernel_rotates[9];
  float kernel_scales[3];

  for (int i = 0; i < 3; i++){
    kernel_centers[i] = centers[batch_id * kernel_size * 3 + thread_id * 3 + i];
  }
  for (int i = 0; i < 9; i++){
    kernel_rotates[i] = rotates[batch_id * kernel_size * 9 + thread_id * 9 + i];
  }
  for (int i = 0; i < 3; i++){
    kernel_scales[i] = scales[batch_id * kernel_size * 3 + thread_id * 3 + i];
  }

  for (int i = 0; (block_id * BLOCK_SIZE + i < query_size) && (i < BLOCK_SIZE); i++) {
    float diff[3];
    float diff_r[3];
    diff[0] = shared_block[i * 3 + 0] - kernel_centers[0];
    diff[1] = shared_block[i * 3 + 1] - kernel_centers[1];
    diff[2] = shared_block[i * 3 + 2] - kernel_centers[2];

    diff_r[0] = kernel_rotates[0] * diff[0] + kernel_rotates[1] * diff[1] + kernel_rotates[2] * diff[2];
    diff_r[1] = kernel_rotates[3] * diff[0] + kernel_rotates[4] * diff[1] + kernel_rotates[5] * diff[2];
    diff_r[2] = kernel_rotates[6] * diff[0] + kernel_rotates[7] * diff[1] + kernel_rotates[8] * diff[2];

    diff_r[0] = diff_r[0] * diff_r[0] * kernel_scales[0];
    diff_r[1] = diff_r[1] * diff_r[1] * kernel_scales[1];
    diff_r[2] = diff_r[2] * diff_r[2] * kernel_scales[2];

    output[output_id_start + i * kernel_size + thread_id] = expf(-(diff_r[0] + diff_r[1] + diff_r[2]));
  }
}

void compute_rbf_cuda(int batch_size, int query_size, int kernel_size,
  const float *points, const float *centers, const float *rotates, const float *scales, float *output) {

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 dimBlock(kernel_size);
  dim3 dimGrid(batch_size, (query_size + kernel_size - 1) / kernel_size);
  compute_rbf_kernel
      <<<dimGrid, dimBlock, kernel_size * 15, stream>>>
        (batch_size, query_size, kernel_size, points, centers, rotates, scales, output);

  CUDA_CHECK_ERRORS();
}
