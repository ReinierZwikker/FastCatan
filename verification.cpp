#include <iostream>
#include <mutex>
#include <cuda_runtime.h>

#include "app/app.h"
#include "src/game/game.h"

bool verify_bean_nn_forward_propagation() {
  #include "src/game/AIPlayer/ai_bean_player.h"

  auto* bean_nn = new BeanNN(false, 0);

  float input[BeanNN::input_nodes];

  clock_t begin_clock = clock();
  for (int i = 0; i < 1; ++i) {
    for (int i = 0; i < BeanNN::input_nodes; ++i) {
      input[i] = (float)i * 1e-10;
    }

    float* output;
    output = bean_nn->calculate_move_probability(input);

    for (int i = 0; i < BeanNN::output_nodes; ++i) {
      std::cout << i << ": " << output[i] << std::endl;
    }
  }
  clock_t end_clock = clock();
  double run_speed = (double)(end_clock - begin_clock) / CLOCKS_PER_SEC;
  std::cout << "BeanNN in " << run_speed * 1000 << " ms" << std::endl;


  delete bean_nn;
  bean_nn = new BeanNN(true, 0);

  cudaStream_t cuda_stream;
  cudaError_t err = cudaStreamCreate(&cuda_stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to create CUDA stream: %s\n", cudaGetErrorString(err));
  }

  clock_t begin_clock_ai = clock();
  for (int i = 0; i < 1; ++i) {
    float input[BeanNN::input_nodes];
    for (int i = 0; i < BeanNN::input_nodes; ++i) {
      input[i] = (float)i * 1e-10;
    }

    float* output_ai;
    output_ai = bean_nn->calculate_move_probability(input, &cuda_stream);

    for (int i = 0; i < BeanNN::output_nodes; ++i) {
      std::cout << i << ": " << output_ai[i] << std::endl;
    }
  }
  clock_t end_clock_ai = clock();
  double run_speed_ai = (double)(end_clock_ai - begin_clock_ai) / CLOCKS_PER_SEC;
  std::cout << "BeanNN CUDA in " << run_speed_ai * 1000 << " ms" << std::endl;



  return true;
}


int main(int argc, char *argv[]) {

  int amount_of_players = atoi(argv[1]);

  verify_bean_nn_forward_propagation();

  return 0;
}