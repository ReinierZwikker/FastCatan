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


Move HumanPlayer::get_move_gui(Board *board, int cards[5]) {
  return player->available_moves[0];
}


Move HumanPlayer::get_move(Board *board, int cards[5]) {
  Move selected_move;

  player_print("Cards:\n");

  for (int card_i = 0; card_i < 5; ++card_i) {
    player_print("    " + card_name(index_card(card_i)) + " = " + std::to_string(cards[card_i]) + "\n");
  }

  player_print("Possible moves:\n");

  int move_i;

  for (move_i = 0; move_i < max_available_moves; ++move_i) {
    if (player->available_moves[move_i].move_type == NoMove) {
      break;
    }
    player_print("Move " + std::to_string(move_i + 1) + ": " + move2string(player->available_moves[move_i]) + "\n");
  }

  int selected_move_i = 0;
  bool valid_selection = false;
  while (!valid_selection) {
    player_print("Select a move:");
    std::cin >> selected_move_i;
    selected_move_i--;
    if (selected_move_i >= 0 && selected_move_i < move_i) {
      valid_selection = true;
    } else {
      player_print("Invalid selection!\n");
    }
  }
  //  selected_move.move_type = index_move(selected_movetype_int);

  selected_move = player->available_moves[selected_move_i];

  player_print("\nSelected move:\n" + move2string(selected_move) + "\n");
  return selected_move;
}

void HumanPlayer::finish_round(Board *board) {

}

HumanPlayer::~HumanPlayer() {
  printf("Thank you for playing!\n");
}
