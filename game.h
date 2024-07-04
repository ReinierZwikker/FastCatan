#ifndef FASTCATAN_GAME_H
#define FASTCATAN_GAME_H

#include "player.h"
#include "board.h"

struct Game {

    Game(int num_players);

    int num_players;
    Player players[6];

    Board board;

    int current_round = 0;

    void start_round();

    void step_round();
};

#endif //FASTCATAN_GAME_H
