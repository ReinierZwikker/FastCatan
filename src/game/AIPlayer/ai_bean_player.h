#ifndef FASTCATAN_BEAN_PLAYER_H
#define FASTCATAN_BEAN_PLAYER_H

#include "../player.h"
#include <cuda_runtime.h>

int test();

class BeanNN {
public:
  BeanNN(bool cuda, unsigned int input_seed);
  BeanNN(const BeanNN* parent_1, const BeanNN* parent_2, const BeanNN* original);
  ~BeanNN();

  // Score parameters
  const float average_points_mult = 1;
  const float win_rate_mult = 20;
  const float average_moves_mult = 1;
  const float mistake_mult = 0.05;

  void calculate_score();

  float* calculate_move_probability(float* input, cudaStream_t* cuda_stream);
  float* calculate_move_probability(const float* input);
  const static uint16_t input_nodes = 240; // 240
  const static uint16_t output_nodes = 10 + 72 + 5 + 5; // 10 + 72 + 5 + 5;
  const static uint8_t num_hidden_layers = 4;
  const static uint16_t nodes_per_layer = 500;

  unsigned int seed;

  float* weights;
  float* biases;

  int weight_size = 0;
  int bias_size = 0;

  PlayerSummary summary{};

private:
  std::mt19937 gen;

  bool cuda_active = false;

  // CUDA memory
  float* device_weights;
  float* device_biases;
  float* device_input;
  float* device_output;
  float* device_layer_1;
  float* device_layer_2;
  float* host_output;

  int get_weight_id(uint8_t connection, uint8_t node_in, uint8_t node_out) const;
  int get_bias_id(uint8_t connection, uint8_t node_out) const;
  static float relu(float value);
};

class BeanPlayer : public PlayerAgent {
public:
  explicit BeanPlayer(Player *connected_player, unsigned int seed);
  Move get_move(Board *board, int cards[5], GameInfo game_info) override;
  void finish_round(Board *board) override;
  inline void unpause(Move move) override {};
  void add_cuda(cudaStream_t* cuda_stream) override;

  void player_print(std::string text);

  inline PlayerType get_player_type() override { return player_type; }
  inline PlayerState get_player_state() override { return player_state; }

  ~BeanPlayer();

  BeanNN* neural_net;
  cudaStream_t* cuda_stream = nullptr;
  bool cuda = false;

private:
  void go_through_moves(MoveType move_type, uint16_t index, CardType tx_card, CardType rx_card);
  void bubble_sort_indices(int* indices_array, const float* array, int size);

  Move chosen_move;

  Player *player;
  const PlayerType player_type = PlayerType::beanPlayer;
  const PlayerState player_state = Waiting;
  std::string console_tag;
};

#endif //FASTCATAN_BEAN_PLAYER_H
