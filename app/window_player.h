#ifndef FASTCATAN_WINDOW_PLAYER_H
#define FASTCATAN_WINDOW_PLAYER_H

#include "../app/3rd_party/imgui/imgui.h"
#include "../app/viewport.h"
#include "../src/game/game.h"
#include "../src/game/components.h"

void CheckAvailableTypes(Game*, int);
void WindowPlayer(Game*, ViewPort*, int);

#endif //FASTCATAN_WINDOW_PLAYER_H
