#ifndef FASTCATAN_WINDOW_AI_H
#define FASTCATAN_WINDOW_AI_H

#include "../app/3rd_party/imgui/imgui.h"
#include "../app/viewport.h"
#include "../src/game/game_manager.h"
#include "../src/game/components.h"

#include <thread>
#include <iostream>
#include <mutex>


class WindowAI {
  public:
    WindowAI();
    ~WindowAI();

    bool show();

  private:
    bool do_training = false;
    bool training = false;

    const unsigned int processor_count = std::thread::hardware_concurrency();
    int num_threads = 30;

    int games_played[30];

    // TODO : Make size depend on processor_count
    GameManager game_managers[30];
    std::thread threads[30];

    // Logging
    int log_type = 0;
};


#endif //FASTCATAN_WINDOW_AI_H
