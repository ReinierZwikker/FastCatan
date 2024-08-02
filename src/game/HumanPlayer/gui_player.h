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
  explicit GuiPlayer(Player *connected_player, std::mutex *mutex, std::condition_variable *con_var, bool *received);
  Move get_move(Board *board, int cards[5]) override;
  void finish_round(Board *board) override;
  ~GuiPlayer();

  std::mutex *waiting;
  std::condition_variable *cv;
  bool *input_received;

  char* tag[20];
  PlayerStates player_state = PlayerStates::Waiting;

private:
  Player *player;
};


#endif //IMGUIAPP_GUI_PLAYER_H
