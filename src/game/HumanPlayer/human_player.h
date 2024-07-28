#ifndef FASTCATAN_HUMAN_PLAYER_H
#define FASTCATAN_HUMAN_PLAYER_H

#include "../components.h"
#include "../board.h"
#include "../player.h"

#include <string>


class HumanPlayer : public PlayerAgent {
public:
  explicit HumanPlayer(Player *connected_player);
  Move get_move(Board *board, int cards[5]) override;
  void finish_round(Board *board) override;
  void player_print(std::string text);
  ~HumanPlayer();

private:
  Player *player;
  std::string console_tag;
};



#endif //FASTCATAN_HUMAN_PLAYER_H
