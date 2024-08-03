#include <cstdio>
#include <iostream>

#include "ai_zwik_player.h"

AIZwikPlayer::AIZwikPlayer(Player *connected_player) {
  player = connected_player;
  console_tag = color_name(connected_player->player_color) + "> " + color_offset(connected_player->player_color);
  player_print("Hello World! I am player number " + std::to_string(color_index(player->player_color) + 1) + "!\n");
}

void AIZwikPlayer::player_print(std::string text) {
  printf("%s%s", console_tag.c_str(), text.c_str());
}

Move AIZwikPlayer::get_move(Board *board, int cards[5]) {
  Move selected_move;

  player_print("My Cards:\n");

  for (int card_i = 0; card_i < 5; ++card_i) {
    player_print("    " + card_name(index_card(card_i)) + " = " + std::to_string(cards[card_i]) + "\n");
  }

  player_print("My possible moves:\n");

  int move_i;

  for (move_i = 0; move_i < max_available_moves; ++move_i) {
    if (player->available_moves[move_i].move_type == NoMove) {
      break;
    }
    player_print("Move " + std::to_string(move_i + 1) + ": " + move2string(player->available_moves[move_i]) + "\n");
  }


  std::mt19937 gen(randomDevice());
  std::uniform_int_distribution<> random_move(0, move_i-1);

  int selected_move_i = random_move(gen);

  selected_move = player->available_moves[selected_move_i];

  player_print("Selecting a random move: " + std::to_string(selected_move_i) + "\n");


  player_print("\nSelected move: " + move2string(selected_move) + "\n");
  return selected_move;
}

void AIZwikPlayer::finish_round(Board *board) {

}

AIZwikPlayer::~AIZwikPlayer() {
  printf("Thank you for playing!\n");
}
