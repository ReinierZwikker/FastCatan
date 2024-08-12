#ifndef FASTCATAN_CONSOLE_PLAYER_H
#define FASTCATAN_CONSOLE_PLAYER_H

#include "../components.h"
#include "../board.h"
#include "../player.h"

#include <string>


class ConsolePlayer : public PlayerAgent {
public:
  explicit ConsolePlayer(Player *connected_player);
  Move get_move(Board *board, int cards[5], GameInfo game_info) override;
  void finish_round(Board *board) override;
  inline void unpause(Move move) override {};

  void player_print(std::string text);

  inline PlayerType get_player_type() override { return player_type; }
  inline PlayerState get_player_state() override { return player_state; }

  ~ConsolePlayer();

private:
  Player *player;
  const PlayerType player_type = PlayerType::consolePlayer;
  const PlayerState player_state = Waiting;
  std::string console_tag;
};


#endif //FASTCATAN_CONSOLE_PLAYER_H
