//
// Created by reini on 25/04/2024.
//

#ifndef FASTCATAN_GAME_H
#define FASTCATAN_GAME_H

#include "player.h"

struct Game {

    Game(int num_players);

    int num_players;
    Player players[4];

    void start_round();

    void step_round();
};

#endif //FASTCATAN_GAME_H
