#ifndef FASTCATAN_PLAYER_H
#define FASTCATAN_PLAYER_H

enum cards {
  Brick,
  Lumber,
  Ore,
  Grain,
  Wool
};

struct Player {
public:
    Player();
    bool activated = false;

    int cards[5]{};
};

#endif //FASTCATAN_PLAYER_H
