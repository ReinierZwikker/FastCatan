#include "window_game.h"

void WindowGame(Game* game) {
  if (ImGui::Button("Start Game")) {
    game->start_game();
  }

  if (ImGui::Button("Step Round")) {
    game->step_round();
  }
}
