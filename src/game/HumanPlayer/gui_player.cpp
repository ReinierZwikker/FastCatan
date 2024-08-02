#include "gui_player.h"

GuiPlayer::GuiPlayer(Player *connected_player, std::mutex *mutex, std::condition_variable *con_var, bool *received) {
  player = connected_player;

  // Mutex system
  waiting = mutex;
  cv = con_var;
  input_received = received;
}

Move GuiPlayer::get_move(Board *board, int cards[5]) {
  Move selected_move;

  std::unique_lock<std::mutex> lock(*waiting);

  player_state = PlayerStates::Playing;

  cv->wait(lock, [this] { return input_received; });

}

void GuiPlayer::finish_round(Board *board) {

}

GuiPlayer::~GuiPlayer() {

}