#ifndef FASTCATAN_RANDOM_PLAYER
#define FASTCATAN_RANDOM_PLAYER

#include "../components.h"
#include "../board.h"
#include "../player.h"

#include <random>
#include <string>


class RandomPlayer : public PlayerAgent {
public:
  explicit RandomPlayer(Player *connected_player);
  Move get_move(Board *board, int cards[5], GameInfo game_info) override;
  void finish_round(Board *board) override;
  inline void unpause(Move move) override {};

  void player_print(std::string text);

  inline PlayerType get_player_type() override { return player_type; }
  inline PlayerState get_player_state() override { return player_state; }

  std::random_device randomDevice;

  ~RandomPlayer();

private:
  Player *player;
  const PlayerType player_type = PlayerType::randomPlayer;
  const PlayerState player_state = Waiting;
  std::string console_tag;

  std::mt19937 gen;
};


#endif //FASTCATAN_RANDOM_PLAYER
