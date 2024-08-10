#ifndef IMGUIAPP_GUI_PLAYER_H
#define IMGUIAPP_GUI_PLAYER_H

#include "../components.h"
#include "../board.h"
#include "../player.h"

#include <mutex>
#include <condition_variable>

#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

class GuiPlayer : public PlayerAgent {
public:
  explicit GuiPlayer(Player *connected_player);
  Move get_move(Board *board, int cards[5]) override;
  void finish_round(Board *board) override;

  inline PlayerType get_player_type() override { return player_type; }
  inline PlayerState get_player_state() override { return player_state; }

  ~GuiPlayer();

  std::mutex waiting;
  std::condition_variable cv;
  bool input_received;
  void unpause(Move move) override;

  char* tag[20];
  PlayerState player_state = PlayerState::Waiting;

private:
  Move selected_move;
  Player *player;
  const PlayerType player_type = guiPlayer;

  static Move load_turn(int move_id);

};


#endif //IMGUIAPP_GUI_PLAYER_H
