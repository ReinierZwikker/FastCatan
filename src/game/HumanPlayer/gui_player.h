#ifndef IMGUIAPP_GUI_PLAYER_H
#define IMGUIAPP_GUI_PLAYER_H

#include "../components.h"
#include "../board.h"
#include "../player.h"

#include <mutex>
#include <condition_variable>

#include <string>

enum PlayerStates {
  Waiting,
  Playing,
  Finished
};

static const char* player_states[] = {
    "Waiting",
    "Playing",
    "Thanks for playing!"
};

class GuiPlayer : public PlayerAgent {
public:
  explicit GuiPlayer(Player *connected_player);
  Move get_move(Board *board, int cards[5]) override;
  void finish_round(Board *board) override;

  inline PlayerType get_player_type() override { return player_type; }

  ~GuiPlayer();

  std::mutex waiting;
  std::condition_variable cv;
  bool input_received;
  void human_input_received();

  char* tag[20];
  PlayerStates player_state = PlayerStates::Waiting;

private:
  Player *player;
  const PlayerType player_type = consolePlayer;
};


#endif //IMGUIAPP_GUI_PLAYER_H
