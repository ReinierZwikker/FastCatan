#ifndef FASTCATAN_WINDOW_COMPONENTS_H
#define FASTCATAN_WINDOW_COMPONENTS_H

#include "../src/game/components.h"

enum class AppState : uint8_t {
  Idle,
  Replaying,
  Training
};

enum class WindowName : uint8_t {
  None,
  Board,
  Player,
  Game,
  AI,
  Replay
};

struct ErrorMessage {
  WindowName window = WindowName::None;
  uint8_t id = 0;
  std::string message;
};

struct AppInfo {
  AppState state = AppState::Idle;
  PlayerType selected_players[4] = {PlayerType::randomPlayer,
                                    PlayerType::randomPlayer,
                                    PlayerType::randomPlayer,
                                    PlayerType::randomPlayer};
  uint8_t num_players = 0;
};

#endif //FASTCATAN_WINDOW_COMPONENTS_H
