#ifndef FASTCATAN_HUMAN_PLAYER_H
#define FASTCATAN_HUMAN_PLAYER_H

#include "../components.h"
#include "../board.h"
#include "../player.h"

#include "NeuralWeb.h"

#include <random>
#include <string>



class AIZwikPlayer : public PlayerAgent {
public:
  explicit AIZwikPlayer(Player *connected_player);
  AIZwikPlayer(Player *connected_player, const std::string& ai_str);

  void player_print(const std::string& text);

  inline PlayerType get_player_type() override { return player_type; }
  inline PlayerState get_player_state() override { return player_state; }
  inline void *get_custom_player_attribute(int attribute_id) override {
    switch (attribute_id) {
      case 0:
        return &neural_web;
      case 1:
        return &mistakes_made;
      default:
        return nullptr;
    }
  }

  ~AIZwikPlayer() override;

  const static int amount_of_neurons = 1500;
  const static int amount_of_env_inputs = 152;
  const static int amount_of_inputs = 608;
  const static int amount_of_outputs = 165;

private:
  Player *player;
  const PlayerType player_type = PlayerType::zwikPlayer;
  const PlayerState player_state = Waiting;
  std::string console_tag;

  float inputs[amount_of_env_inputs + amount_of_inputs] = {};
  float outputs[amount_of_outputs] = {};

  NeuralWeb neural_web;

public:
    void update_environment();
    Move get_move(Board *board, int cards[5], GameInfo game_info) override;
    void finish_round(Board *board) override;
    inline void unpause(Move move) override {};

    int mistakes_made;
};


#endif //FASTCATAN_HUMAN_PLAYER_H
