#include <cstdio>
#include <iostream>

#include "random_player.h"

RandomPlayer::RandomPlayer(Player *connected_player, unsigned int input_seed) : gen(input_seed) {
  player = connected_player;
  console_tag = color_name(connected_player->player_color) + "> " + color_offset(connected_player->player_color);
  // player_print("Hello World! I am player number " + std::to_string(color_index(player->player_color) + 1) + "!\n");
}

void RandomPlayer::player_print(std::string text) {
  printf("%s%s", console_tag.c_str(), text.c_str());
}

Move RandomPlayer::get_move(Board *board, int cards[5], GameInfo game_info) {
  Move selected_move;

//  player_print("My Cards:\n");
//  for (int card_i = 0; card_i < 5; ++card_i) {
//    player_print("    " + card_name(index_card(card_i)) + " = " + std::to_string(cards[card_i]) + "\n");
//  }
//  player_print("My possible moves:\n");

  int move_i;

  for (move_i = 0; move_i < max_available_moves; ++move_i) {
    if (player->available_moves[move_i].type == MoveType::NoMove) {
      break;
    }
//    player_print("Move " + std::to_string(move_i + 1) + ": " + move2string(player->available_moves[move_i]) + "\n");
  }


  int selected_move_i;

  if (move_i <= 1) {
    selected_move_i = 0;
  } else {
    std::uniform_int_distribution<> random_move(0, move_i-1);
    selected_move_i = random_move(gen);

  }
  selected_move = player->available_moves[selected_move_i];

//  player_print("Selecting a random move: " + std::to_string(selected_move_i) + "\n");
//  player_print("\nSelected move: " + move2string(selected_move) + "\n");
  return selected_move;
}

void RandomPlayer::finish_round(Board *board) {

}

RandomPlayer::~RandomPlayer() = default;
