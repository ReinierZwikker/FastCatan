#include <cuda_runtime.h>

void step_feed_forward(const float *input_array, const float *weights, const float *biases,
                       int matrix_rows, int matrix_columns, float *output_array, cudaStream_t* stream);

void malloc_weights_biases(const float *weights, const float *biases, int rows, int columns);
