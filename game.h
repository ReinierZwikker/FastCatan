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

    void start_game();

    void step_round();

    void give_cards(int rolled_number);
};

#endif //FASTCATAN_GAME_H
