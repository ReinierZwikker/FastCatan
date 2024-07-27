#ifndef FASTCATAN_HUMAN_PLAYER_H
#define FASTCATAN_HUMAN_PLAYER_H

#include "../components.h"
#include "../board.h"
#include "../player.h"

class HumanPlayer : public Agent {
public:
  HumanPlayer(int assigned_player_number);
  Move get_move(Board *board, int cards[5]);
  void finish_round(Board *board);
  ~HumanPlayer();

private:
  int player_number;
};



#endif //FASTCATAN_HUMAN_PLAYER_H
