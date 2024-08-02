#include "gui_player.h"
#include "iostream"

GuiPlayer::GuiPlayer(Player *connected_player) {
  player = connected_player;
}

Move GuiPlayer::get_move(Board *board, int cards[5]) {
  Move selected_move;

  std::cout << "test" << std::endl;
  std::unique_lock<std::mutex> lock(waiting);

  player_state = PlayerStates::Playing;

  cv.wait(lock, [this] { return input_received; });

  player_state = PlayerStates::Waiting;

  return selected_move;
}

void GuiPlayer::human_input_received() {

}

void GuiPlayer::finish_round(Board *board) {

}

GuiPlayer::~GuiPlayer() {

}