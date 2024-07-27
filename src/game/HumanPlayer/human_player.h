#ifndef FASTCATAN_HUMAN_PLAYER_H
#define FASTCATAN_HUMAN_PLAYER_H

#include "../components.h"
#include "../board.h"
#include "../player.h"

class HumanPlayer : public PlayerAgent {
public:
  explicit HumanPlayer(Player *connected_player);
  Move get_move(Board *board, int cards[5]) override;
  void finish_round(Board *board) override;
  ~HumanPlayer();

private:
  Player *player;
};



#endif //FASTCATAN_HUMAN_PLAYER_H
