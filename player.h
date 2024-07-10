#ifndef FASTCATAN_PLAYER_H
#define FASTCATAN_PLAYER_H



struct Player {
public:
    Player();
    bool activated = false;



    int cards[5]{};
};

#endif //FASTCATAN_PLAYER_H
