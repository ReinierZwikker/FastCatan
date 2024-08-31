#ifndef FASTCATAN_WINDOW_AI_H
#define FASTCATAN_WINDOW_AI_H

#include "imgui.h"
#include "viewport.h"
#include "game/game_manager.h"
#include "game/components.h"
#include "app_components.h"

#include <thread>
#include <iostream>
#include <mutex>


class WindowAI {
public:
  WindowAI();
  ~WindowAI();

  inline void show(Game* game, AppInfo* app_info) {};
};


#endif //FASTCATAN_WINDOW_AI_H
