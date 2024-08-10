#include "gui_player.h"

GuiPlayer::GuiPlayer(Player *connected_player) {
  player = connected_player;
  input_received = false;
}

Move GuiPlayer::get_move(Board *board, int cards[5]) {
  std::unique_lock<std::mutex> lock(waiting);

  player_state = PlayerState::Playing;

  cv.wait(lock, [this] { return input_received; });
  input_received = false;

  if (selected_move.type == MoveType::Replay) {
//    Move move = load_turn(selected_move.index);
//    selected_move = move;
  }

  player_state = PlayerState::Waiting;

  return selected_move;
}

void GuiPlayer::unpause(Move move) {
  std::lock_guard<std::mutex> lock(waiting);
  selected_move = move;
  input_received = true;
  cv.notify_one();
}

//Move GuiPlayer::load_turn(int move_id) {
//  Move move{};
//  if (!std::filesystem::exists("replay")) {
//    FILE* file = std::fopen("replay/GameReplay", "rb");
//
//    if (!file) {
//      throw std::invalid_argument("Could not open replay/GameReplay.txt");
//    }
//
//    // Find the size of the file
//    fseek(file, 0, SEEK_END);
//    long file_size = ftell(file);
//    fseek(file, 0, SEEK_SET);
//
//    size_t amount_of_moves = file_size / sizeof(Move);
//
//    // Allocate memory for the moves
//    Move* moves = (Move*)malloc(file_size);
//    if (!moves) {
//      fclose(file);
//      throw std::invalid_argument("The memory allocation for the moves of the gui player has failed");
//    }
//
//    std::fread(&moves, sizeof(Move), amount_of_moves, file);
//    move = moves[move_id];
//
//    fclose(file);
//  }
//
//  return move;
//}

void GuiPlayer::finish_round(Board *board) {

}

GuiPlayer::~GuiPlayer() {

}