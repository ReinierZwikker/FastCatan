#ifndef IMGUIAPP_WINDOW_REPLAY_H
#define IMGUIAPP_WINDOW_REPLAY_H

#include "../app/3rd_party/imgui/imgui.h"
#include "../app/viewport.h"
#include "../src/game/game_manager.h"
#include "../src/game/components.h"

#include <thread>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#include <mutex>

class WindowReplay {
  public:
    WindowReplay();
    ~WindowReplay();

    void show();

  private:

    void load_games(const std::string& folder);
    void transfer(const std::string& folder, int game_id);

    bool invalid_input_folder = false;
    int game_number = 0;
    int thread_id = 1;
    std::vector<GameSummary> loaded_games;
    std::vector<Move> loaded_moves;
};

#endif //IMGUIAPP_WINDOW_REPLAY_H
