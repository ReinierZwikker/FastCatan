#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void forward_atomics_kernel(const float *input_array,
                                       const float *weights,
                                       const float *biases,
                                       int input_size,   // N
                                       int matrix_size,  // MxN
                                       float *output_array) {

  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < matrix_size) {
    auto in_id = id % input_size;
    auto out_id = id / input_size;
    auto value = input_array[in_id] * weights[id];
    if (in_id == 0) {
      value += biases[out_id];
    }
    atomicAdd(&output_array[out_id], value);
  }
}

__global__ void forward_kernel(const float *input_array,
                               const float *weights,
                               const float *biases,
                               int input_size,   // N
                               int output_size,  // M
                               float *output_array) {

  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < output_size) {
    float value = 0.0f;
    for (int column = 0; column < input_size; ++column) {
      value += weights[row * input_size + column] * input_array[column];
    }

    // Bias
    value += biases[row];

    // ReLu
    if (value < 0) {
      output_array[row] = 0;
    }
    else {
      output_array[row] = value;
    }
  }
}

// Wrapper function to be called from C++
void step_feed_forward(const float *input_array, const float *weights, const float *biases,
                       int matrix_rows, int matrix_columns, float *output_array, cudaStream_t* stream) {
//  const int matrix_size = matrix_rows * matrix_columns;
//  const int numBlocks = (matrix_size + blockSize - 1) / blockSize;
//
//  forward_atomics_kernel<<<numBlocks, blockSize, 0, *stream>>>(input_array, weights, biases,
//                                                               matrix_columns, matrix_size, output_array);
//
  const int blockSize = 256;
  const int numBlocks = (matrix_rows + blockSize - 1) / blockSize;

  forward_kernel<<<numBlocks, blockSize, 0, *stream>>>(input_array, weights, biases,
                                                       matrix_columns, matrix_rows, output_array);

  // cudaDeviceSynchronize();

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  }
}