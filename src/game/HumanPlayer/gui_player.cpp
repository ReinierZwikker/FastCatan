#include "gui_player.h"
#include "iostream"
#include <thread>

GuiPlayer::GuiPlayer(Player *connected_player) {
  player = connected_player;
  input_received = false;
}

Move GuiPlayer::get_move(Board *board, int cards[5]) {
  std::unique_lock<std::mutex> lock(waiting);

  player_state = PlayerState::Playing;

  cv.wait(lock, [this] { return input_received; });
  input_received = false;

  std::cout << "[get_move] unlocked" << std::endl;

  player_state = PlayerState::Waiting;

  return selected_move;
}

void GuiPlayer::unpause(Move move) {
  std::lock_guard<std::mutex> lock(waiting);
  selected_move = move;
  input_received = true;
  cv.notify_one();
}

void GuiPlayer::finish_round(Board *board) {

}

GuiPlayer::~GuiPlayer() {

}