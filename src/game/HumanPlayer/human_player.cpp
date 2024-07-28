#include <cstdio>
#include <iostream>

#include "human_player.h"

HumanPlayer::HumanPlayer(Player *connected_player) {
  player = connected_player;
  console_tag = color_name(connected_player->player_color) + "> " + color_offset(connected_player->player_color);
  player_print("Welcome to FastCatan! You are player number " + std::to_string(color_index(player->player_color) + 1) + "!\n");
}

void HumanPlayer::player_print(std::string text) {
  printf("%s%s", console_tag.c_str(), text.c_str());
}


Move HumanPlayer::get_move(Board *board, int cards[5]) {
  Move selected_move;

  player_print("Select a move:");
  int selected_movetype_int = 0;
  std::cin >> selected_movetype_int;
  selected_move.move_type = index_move(selected_movetype_int);

  player_print(move2string(selected_move) + "\n");
  return selected_move;
}

void HumanPlayer::finish_round(Board *board) {

}

HumanPlayer::~HumanPlayer() {
  printf("Thank you for playing!\n");
}
