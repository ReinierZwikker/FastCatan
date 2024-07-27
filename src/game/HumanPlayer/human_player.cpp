#include <cstdio>
#include "human_player.h"

HumanPlayer::HumanPlayer(Player *connected_player) {
  player = connected_player;
  printf("\n\nWelcome to FastCatan! You are player number %d!\n", color_index(player->player_color));
}


Move HumanPlayer::get_move(Board *board, int cards[5]) {
  return {};
}

void HumanPlayer::finish_round(Board *board) {

}

HumanPlayer::~HumanPlayer() {
  printf("Thank you for playing!\n");
}
