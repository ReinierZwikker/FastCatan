#ifndef FASTCATAN_HUMAN_PLAYER_H
#define FASTCATAN_HUMAN_PLAYER_H

#include "../components.h"
#include "../board.h"
#include "../player.h"

#include <string>


class ConsolePlayer : public PlayerAgent {
public:
  explicit ConsolePlayer(Player *connected_player);
  Move get_move(Board *board, int cards[5]) override;
  void finish_round(Board *board) override;

  void player_print(std::string text);

  inline PlayerType get_player_type() override { return player_type; }

  ~ConsolePlayer();

private:
  Player *player;
  const PlayerType player_type = consolePlayer;
  std::string console_tag;
};


#endif //FASTCATAN_HUMAN_PLAYER_H
