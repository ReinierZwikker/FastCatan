#ifndef FASTCATAN_BEAN_PLAYER_H
#define FASTCATAN_BEAN_PLAYER_H

#include "../player.h"

class BeanNN {
public:
  BeanNN(uint8_t num_layers, uint16_t nodes_per_row, unsigned int input_seed);
  ~BeanNN();

  float* calculate_move_probability(float* input);
  const static uint16_t input_nodes = 240;
  const static uint16_t output_nodes = 10 + 72 + 5 + 5;

  unsigned int seed;

private:
  float* weights;
  float* biases;
  std::mt19937 gen;

  uint8_t num_hidden_layers = 0;
  uint16_t nodes_per_layer = 0;

  int weight_size = 0;
  int bias_size = 0;

  int get_weight_id(uint8_t connection, uint8_t node_in, uint8_t node_out) const;
  static float relu(float value);
};

class BeanPlayer : public PlayerAgent {
public:
  explicit BeanPlayer(Player *connected_player, unsigned int seed);
  Move get_move(Board *board, int cards[5], GameInfo game_info) override;
  void finish_round(Board *board) override;
  inline void unpause(Move move) override {};

  void player_print(std::string text);

  inline PlayerType get_player_type() override { return player_type; }
  inline PlayerState get_player_state() override { return player_state; }

  ~BeanPlayer();

  BeanNN* neural_net;

private:
  Move go_through_moves(MoveType move_type, uint16_t index, CardType tx_card, CardType rx_card);
  void bubble_sort_indices(int* indices_array, const float* array, int size);

  Player *player;
  const PlayerType player_type = PlayerType::beanPlayer;
  const PlayerState player_state = Waiting;
  std::string console_tag;
};

#endif //FASTCATAN_BEAN_PLAYER_H
