#ifndef FASTCATAN_WINDOW_PLAYER_H
#define FASTCATAN_WINDOW_PLAYER_H

#include "imgui.h"
#include "viewport.h"
#include "game/game.h"
#include "game/components.h"
#include "app_components.h"

void CheckAvailableTypes(Game*, int);
void WindowPlayer(Game*, ViewPort*, int, AppInfo*);

#endif //FASTCATAN_WINDOW_PLAYER_H
