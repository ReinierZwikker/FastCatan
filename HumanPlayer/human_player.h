#ifndef FASTCATAN_HUMAN_PLAYER_H
#define FASTCATAN_HUMAN_PLAYER_H

#include "../components.h"
#include "../board.h"
#include "../player.h"

class HumanPlayer : public Agent {
public:
  Move get_move(Board *board, int cards[5]);
  void finish_round(Board *board);


};



#endif //FASTCATAN_HUMAN_PLAYER_H
