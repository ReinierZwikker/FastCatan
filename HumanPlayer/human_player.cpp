#include <cstdio>
#include "human_player.h"

HumanPlayer::HumanPlayer(int assigned_player_number) {
  player_number = assigned_player_number;
  printf("\n\nWelcome to FastCatan! You are player number %d\n", player_number);
}


Move HumanPlayer::get_move(Board *board, int cards[5]) {
  return {};
}

void HumanPlayer::finish_round(Board *board) {

}

HumanPlayer::~HumanPlayer() {
  printf("Thank you for playing!\n");
}
